"""A decision model: an LLM trained to pick one option among a list, in a single forward pass.

OneJev1.6-4B is Qwen3.5-4B fine-tuned by its authors on decision questions. A request lists the options
behind slot labels (A, B, C... then two-letter labels past 26), and the answer is the model's next-token
distribution over those slot tokens, read from the output head in float32. The model ships no calibration
file, so the distribution is used as is (temperature 1), with the options in one order, as the Decision
Index runs it.

The rendering mirrors the model's own serving code (Flide, prompt ``onejev-labels-v2``:
github.com/OmniJev/Flide, ``plugins/onejev/flide_onejev/prompt.py``), and requests are worded as the
Decision Index kit words them for VAST and CLINC150 (github.com/apolinario/decision-index,
``suite/build/normalize_text.py`` and ``normalize_direct.py``): the model was tuned on that interface. The
CLINC150 slices, which the index does not have, drop the out-of-scope clause. DynaSent, also absent
from the index, is worded after its VAST request.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd

from lc import data
from lc.models.prompted import model_disk_mb

SYSTEM_PROMPT = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Two-letter labels that Qwen's tokenizer splits into two tokens, so they cannot be slots.
_NOT_ONE_TOKEN = {
    "BQ",
    "BZ",
    "CJ",
    "CQ",
    "CZ",
    "DQ",
    "DZ",
    "EJ",
    "EY",
    "FJ",
    "FQ",
    "FV",
    "FZ",
    "GJ",
    "GK",
    "GQ",
    "GZ",
    "HJ",
    "IY",
    "JF",
    "JG",
    "JH",
    "JL",
    "JN",
    "JQ",
}
SLOT_LABELS = [
    *LETTERS,
    *[a + b for a in LETTERS for b in LETTERS if a + b not in _NOT_ONE_TOKEN][: 256 - len(LETTERS)],
]
# The Decision Index's VAST options, in its order, with this repository's label names.
VAST_OPTIONS = [("against", "against"), ("favor", "for"), ("neutral", "neutral")]
LABELS_FILE = "labels.json"


def request(task: str, text: str, labels: list[str]) -> tuple[str, list[tuple[str, str]], list[str]]:
    """(instructions, [(option key, description)], our label for each option) for one text."""
    if task == "vast":
        topic, _, comment = text.partition("\nComment: ")
        instructions = (
            f"Topic: {topic.removeprefix('Topic: ')}\nPost: {comment}\n"
            "Determine the stance of the post toward the topic."
        )
        keys = [chr(65 + i) for i in range(len(VAST_OPTIONS))]
        options = list(zip(keys, [option for option, _ in VAST_OPTIONS], strict=True))
        return instructions, options, [label for _, label in VAST_OPTIONS]
    if task == "dynasent":
        instructions = f"Sentence: {text}\nDetermine the sentiment of the sentence."
        keys = [chr(65 + i) for i in range(len(labels))]
        return instructions, list(zip(keys, labels, strict=True)), list(labels)
    if task == data.CLINC150:
        head = "Classify the intent of this user request, or choose out of scope if none applies:\n"
    else:
        head = "Classify the intent of this user request:\n"
    words = [
        "out of scope: none of the listed intents" if label == data.OUT_OF_SCOPE else label.replace("_", " ")
        for label in labels
    ]
    keys = [chr(65 + i) if len(labels) <= 26 else f"option_{i}" for i in range(len(labels))]
    return head + text, list(zip(keys, words, strict=True)), list(labels)


def messages(instructions: str, options: list[tuple[str, str]]) -> list[dict[str, str]]:
    """The chat for one request. The index sends no state, which renders as an empty JSON object."""
    slots = SLOT_LABELS[: len(options)]
    listing = "\n".join(
        f"{slot}. {key}: {text.strip()}" for slot, (key, text) in zip(slots, options, strict=True)
    )
    answer = "Answer with one letter: {}." if len(slots) <= len(LETTERS) else "Answer with one label: {}."
    question = f"Question: {instructions.strip()}\n\nOptions:\n{listing}\n\n" + answer.format(
        ", ".join(slots)
    )
    state = f"<state>\n{json.dumps({})}\n</state>\n\n"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": state + question}]


@dataclass(frozen=True)
class DecisionModel:
    repo: str
    revision: str

    def identity(self) -> dict:
        return {"approach": "decision-model", "prompt": "onejev-labels-v2", **asdict(self)}

    def fit(self, task: str, train: pd.DataFrame, labels: list[str], seed: int, out_dir: Path) -> int:
        """Nothing to train and no example to show: record the task and its labels."""
        (out_dir / LABELS_FILE).write_text(json.dumps({"task": task, "labels": labels}))
        return 0

    def load(self, out_dir: Path) -> "DecisionPredictor":
        spec = json.loads((out_dir / LABELS_FILE).read_text())
        return DecisionPredictor(SlotScorer(self.repo, self.revision), spec["task"], spec["labels"])

    def disk_mb(self, out_dir: Path) -> float:
        return model_disk_mb(self.repo, self.revision)


class DecisionPredictor:
    def __init__(self, scorer: "SlotScorer", task: str, labels: list[str]):
        self.scorer, self.task, self.labels = scorer, task, labels

    def predict(self, text: str) -> np.ndarray:
        instructions, options, order = request(self.task, text, self.labels)
        probs = self.scorer.score(messages(instructions, options), len(options))
        return probs[[order.index(label) for label in self.labels]]


class SlotScorer:
    """Next-token distribution over the slot tokens at the end of a request."""

    def __init__(self, repo: str, revision: str):
        from huggingface_hub import snapshot_download
        from mlx_lm.utils import _get_classes, load_model, load_tokenizer

        def classes(config):
            model_class, args_class = _get_classes(config)

            class TiedHead(model_class):
                def sanitize(self, weights):
                    # The checkpoint also stores the output head, an exact copy of the input embeddings,
                    # which the configuration ties. mlx-lm only drops that key under its inner name.
                    weights.pop("lm_head.weight", None)
                    return super().sanitize(weights)

            return TiedHead, args_class

        path = Path(snapshot_download(repo, revision=revision))
        self.model, _ = load_model(path, get_model_classes=classes)
        self.tokenizer = load_tokenizer(path)
        self.decoder = self.model.language_model.model  # the text decoder, up to its final norm
        ids = [self._single_token(label) for label in SLOT_LABELS]
        assert None not in ids and len(set(ids)) == len(ids), "slot labels must be distinct single tokens"
        self.slot_weights = self.decoder.embed_tokens.weight[mx.array(ids)].astype(mx.float32)

    def _single_token(self, label: str) -> int | None:
        ids = self.tokenizer.encode(label, add_special_tokens=False)
        return ids[0] if len(ids) == 1 and self.tokenizer.decode(ids) == label else None

    def score(self, chat: list[dict[str, str]], n_options: int) -> np.ndarray:
        text = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        hidden = self.decoder(mx.array([ids]))[0, -1].astype(mx.float32)
        return np.array(mx.softmax(self.slot_weights[:n_options] @ hidden).tolist())
