"""A generic LLM, prompted with the list of labels, zero-shot or with a few labeled examples.

The model never generates text. For each label, the label scorer computes the probability that the
assistant's answer is exactly that label, from the model's log-probabilities: every token of the label,
then the end-of-turn token. Normalized over the labels, these give one probability per label, like the
other approaches. The answer is always a valid label, and nothing has to be parsed.

Scoring 151 labels for every text would be slow done naively. Two caches keep it close to the cost of a
single forward pass:

- The instructions, the label list and the few-shot examples come before the text to classify, so every
  prompt of a run starts with the same tokens. That shared prefix is encoded once per run, and each text
  only adds its own tokens.
- The prompt is encoded once per text; its cache is then copied for the labels, which run in batches of
  ``LABEL_CHUNK`` in a single forward pass each.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache

from lc import data
from lc.measure import disk_mb

LABEL_CHUNK = 32
SPEC_FILE = "prompt.json"
# How one labeled example reads in a few-shot prompt, per dataset (VAST texts hold topic and comment).
EXAMPLE_FORMAT = {
    "clinc150": "Message: {text}\nIntent: {label}",
    "vast": "{text}\nStance: {label}",
    "dynasent": "Sentence: {text}\nSentiment: {label}",
}


def readable(label: str) -> str:
    """What the model reads and answers: words, not identifiers (``pto_request`` becomes "pto request")."""
    return "out of scope" if label == data.OUT_OF_SCOPE else label.replace("_", " ")


def render(template: str, text: str, labels: list[str], examples: str = "") -> str:
    return template.format(
        text=text, labels=", ".join(labels), label_lines="\n".join(labels), examples=examples
    )


def render_examples(task: str, examples: pd.DataFrame) -> str:
    line = EXAMPLE_FORMAT[_dataset(task)]
    return "\n\n".join(
        line.format(text=text, label=readable(label))
        for text, label in zip(examples["text"], examples["label"], strict=True)
    )


@dataclass(frozen=True)
class PromptedLLM:
    """``shots`` maps a dataset (``clinc150``, ``vast``) to examples per class; empty for zero-shot."""

    repo: str
    revision: str
    prompt_dir: str
    shots: tuple[tuple[str, int], ...] = ()

    @property
    def prompt_name(self) -> str:
        return "few-shot" if self.shots else "zero-shot"

    def template(self, task: str) -> str:
        return (Path(self.prompt_dir) / _dataset(task) / f"{self.prompt_name}.txt").read_text()

    def identity(self) -> dict:
        templates = sorted(Path(self.prompt_dir).glob(f"*/{self.prompt_name}.txt"))
        digest = hashlib.sha256(b"".join(path.read_bytes() for path in templates)).hexdigest()[:12]
        return {
            "approach": f"prompted-llm-{self.prompt_name}",
            "repo": self.repo,
            "revision": self.revision,
            "prompts": digest,
            "shots": dict(self.shots),
        }

    def fit(self, task: str, train: pd.DataFrame, labels: list[str], seed: int, out_dir: Path) -> int:
        """Nothing to train: render the prompt's fixed part, drawing the few-shot examples if any."""
        per_class = dict(self.shots).get(_dataset(task), 0)
        examples = data.few_shot_examples(task, per_class, seed) if per_class else train.iloc[:0]
        spec = {
            "template": self.template(task),
            "labels": [readable(label) for label in labels],
            "examples": render_examples(task, examples),
        }
        assert len(set(spec["labels"])) == len(labels), "readable label names must stay distinct"
        assert bool(per_class) == ("{examples}" in spec["template"]), "few-shot needs an {examples} slot"
        (out_dir / SPEC_FILE).write_text(json.dumps(spec))
        return len(examples)

    def load(self, out_dir: Path) -> "PromptedPredictor":
        spec = json.loads((out_dir / SPEC_FILE).read_text())
        return PromptedPredictor(LabelScorer(self.repo, self.revision), **spec)

    def disk_mb(self, out_dir: Path) -> float:
        return model_disk_mb(self.repo, self.revision)


class PromptedPredictor:
    def __init__(self, scorer: "LabelScorer", template: str, labels: list[str], examples: str):
        self.scorer, self.template, self.labels, self.examples = scorer, template, labels, examples
        # Two prompts that differ only in their text share exactly the fixed part of the prompt.
        scorer.share_prefix(self._prompt("a"), self._prompt("b"))

    def _prompt(self, text: str) -> str:
        return render(self.template, text, self.labels, self.examples)

    def predict(self, text: str) -> np.ndarray:
        return self.scorer.score(self._prompt(text), self.labels)


class LabelScorer:
    """Probability of each label as the complete answer of the assistant turn (thinking off)."""

    def __init__(self, repo: str, revision: str):
        self.model, self.tokenizer = load(repo, revision=revision)
        self.end_of_turn = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._label_ids: dict[tuple[str, ...], list[list[int]]] = {}
        self._prefix: list[int] = []
        self._prefix_cache = None

    def prompt_ids(self, message: str) -> list[int]:
        """Token ids of a one-turn chat, up to where the assistant's answer starts."""
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": message}], add_generation_prompt=True, enable_thinking=False
        )

    def label_ids(self, labels: list[str]) -> list[list[int]]:
        key = tuple(labels)
        if key not in self._label_ids:
            self._label_ids[key] = [
                self.tokenizer.encode(label, add_special_tokens=False) + [self.end_of_turn]
                for label in labels
            ]
        return self._label_ids[key]

    def share_prefix(self, *messages: str) -> int:
        """Encode once the tokens the messages start with; later prompts starting with them reuse the cache.

        Qwen3.5 mixes attention layers with linear-attention (recurrent) layers, whose state cannot be cut
        back to an earlier position. A prompt whose tokens differ inside the prefix (the text merging with
        the token before it, say) is therefore encoded in full rather than trimmed.
        """
        ids = [self.prompt_ids(m) for m in messages]
        n = _common_prefix_length(ids)
        self._prefix, self._prefix_cache = ids[0][:n], make_prompt_cache(self.model)
        if n:
            mx.eval(self.model(mx.array([self._prefix]), cache=self._prefix_cache))
        return n

    def score(self, message: str, labels: list[str], use_cache: bool = True) -> np.ndarray:
        prompt = self.prompt_ids(message)
        label_ids = self.label_ids(labels)
        logprobs = self._cached(prompt, label_ids) if use_cache else self._uncached(prompt, label_ids)
        probs = np.exp(logprobs - logprobs.max())
        return probs / probs.sum()

    def _cached(self, prompt: list[int], label_ids: list[list[int]]) -> np.ndarray:
        n = len(self._prefix)
        if 0 < n < len(prompt) and prompt[:n] == self._prefix:
            cache, start = _copy_cache(self._prefix_cache), n
        else:
            cache, start = make_prompt_cache(self.model), 0
        last = self.model(mx.array([prompt[start:]]), cache=cache)[0, -1].astype(mx.float32)
        first = np.array((last - mx.logsumexp(last))[mx.array([ids[0] for ids in label_ids])])
        rest = np.zeros(len(label_ids))
        for chunk_start in range(0, len(label_ids), LABEL_CHUNK):
            chunk = label_ids[chunk_start : chunk_start + LABEL_CHUNK]
            width = max(len(ids) for ids in chunk) - 1
            if width == 0:
                continue
            # Each label minus its last token, right-padded: padding comes after the real tokens, so the
            # causal layers never let it reach the positions that are read.
            inputs = np.zeros((len(chunk), width), dtype=np.int32)
            for i, ids in enumerate(chunk):
                inputs[i, : len(ids) - 1] = ids[:-1]
            chunk_cache = _copy_cache(cache)
            _repeat_cache(chunk_cache, len(chunk))
            logits = self.model(mx.array(inputs), cache=chunk_cache).astype(mx.float32)
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            for i, ids in enumerate(chunk):
                picked = logp[i, mx.arange(len(ids) - 1), mx.array(ids[1:])]
                rest[chunk_start + i] = float(picked.sum())
        return first + rest

    def _uncached(self, prompt: list[int], label_ids: list[list[int]]) -> np.ndarray:
        """Reference implementation: one full forward pass per label, no cache."""
        scores = []
        for ids in label_ids:
            logits = self.model(mx.array([prompt + ids[:-1]]))[0, len(prompt) - 1 :].astype(mx.float32)
            logp = np.array(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
            scores.append(sum(logp[j, token] for j, token in enumerate(ids)))
        return np.array(scores)


def model_disk_mb(repo: str, revision: str) -> float:
    """Size of the model files in the local Hugging Face cache (weights, config and tokenizer)."""
    from huggingface_hub import try_to_load_from_cache

    return disk_mb(Path(try_to_load_from_cache(repo, "config.json", revision=revision)).parent)


def _dataset(task: str) -> str:
    return task.split("-")[0]


def _common_prefix_length(sequences: list[list[int]]) -> int:
    n = 0
    for column in zip(*sequences, strict=False):
        if len(set(column)) > 1:
            break
        n += 1
    return n


def _copy_cache(cache: list) -> list:
    """An independent copy of a prompt cache. Key and value arrays are sliced to their length, so the copy
    reallocates on its first update instead of writing into arrays the original shares."""
    copy = [KVCache() if isinstance(c, KVCache) else ArraysCache(len(c.cache)) for c in cache]
    for original, new in zip(cache, copy, strict=True):
        if isinstance(original, KVCache):
            keys, values = original.keys_and_values()
            new.state = keys[..., : original.offset, :], values[..., : original.offset, :], original.offset
        else:
            new.cache = list(original.cache)
    return copy


def _repeat_cache(cache: list, n: int) -> None:
    """Turn a batch-1 cache into a batch-n cache, one copy per label of a chunk."""
    for c in cache:
        if isinstance(c, KVCache):
            keys, values = c.keys_and_values()
            c.state = mx.repeat(keys, n, axis=0), mx.repeat(values, n, axis=0), c.offset
        else:
            c.cache = [None if a is None else mx.repeat(a, n, axis=0) for a in c.cache]
