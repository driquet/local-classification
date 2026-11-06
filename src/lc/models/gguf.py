"""The 4B models served by llama.cpp instead of MLX: same prompts, same scoring, on the GPU or the CPU.

Both scorers mirror their MLX counterparts and plug into the same predictors:

- ``SlotScorer`` (OneJev): one pass over the request, then the next-token distribution over the slot tokens.
- ``LabelScorer`` (Qwen zero-shot): the probability of each label as the complete answer. The shared prefix
  is encoded once, in sequence 0, and never touched. Each text copies it to sequence 1 and adds its own
  tokens; each chunk of labels then copies sequence 1 into one sequence per label and runs in one batch.
  In llama.cpp a copy shares the attention cache and the recurrent state until a sequence writes to it,
  which works for Qwen3.5's linear-attention layers too, whose state cannot be cut back.

The process imports neither torch nor MLX: the chat template is rendered with jinja2, as transformers
renders it, and tokenized with ``tokenizers``. Weights are read into RAM (no mmap), so that the physical
footprint counts them, as ``measure.in_ram`` does for PyTorch on the CPU.
"""

import ctypes
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lc.models.decision import LABELS_FILE, SLOT_LABELS, DecisionModel, DecisionPredictor
from lc.models.prompted import LABEL_CHUNK, PromptedLLM, PromptedPredictor, common_prefix_length
from lc.models.prompted import SPEC_FILE as PROMPT_SPEC

N_CTX = 8192  # the longest prompt plus a chunk of labels; the cache is allocated up front
N_BATCH = 2048  # llama.cpp's defaults, as llama-cli and llama-server use them
N_UBATCH = 512
_QUIET = None  # the log callback that silences llama.cpp


class Chat:
    """A model's chat template and tokenizer, from its Hugging Face files, without transformers."""

    def __init__(self, repo: str, revision: str):
        import jinja2
        from huggingface_hub import snapshot_download
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        path = Path(snapshot_download(repo, revision=revision))
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        env.globals["raise_exception"] = raise_exception
        self.template = env.from_string((path / "chat_template.jinja").read_text())

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def prompt_ids(self, chat: list[dict[str, str]]) -> list[int]:
        """Token ids of a chat, up to where the assistant's answer starts (thinking off)."""
        text = self.template.render(messages=chat, add_generation_prompt=True, enable_thinking=False)
        return self.encode(text)

    def token_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)


class Runtime:
    """A llama.cpp model and context: ``device`` is ``metal`` or ``cpu<threads>``."""

    def __init__(self, gguf: str, device: str, n_seq: int = 1):
        import llama_cpp as ll

        self.ll = ll
        ll.llama_backend_init()
        global _QUIET
        _QUIET = _QUIET or ll.llama_log_callback(lambda *_: None)  # kept alive: C calls it later
        ll.llama_log_set(_QUIET, ctypes.c_void_p(0))
        model_params = ll.llama_model_default_params()
        model_params.n_gpu_layers = -1 if device == "metal" else 0
        model_params.load_mode = ll.LLAMA_LOAD_MODE_NONE  # weights read into RAM, not mapped
        model_params.lazy_mode = ll.LLAMA_LAZY_MODE_OFF
        model_params.load_mtp = False  # MLX skips the multi-token head too; OneJev has none
        self.model = ll.llama_model_load_from_file(gguf.encode(), model_params)
        assert self.model, f"llama.cpp could not load {gguf}"
        threads = 4 if device == "metal" else int(device.removeprefix("cpu"))
        params = ll.llama_context_default_params()
        params.n_ctx, params.n_batch, params.n_ubatch, params.n_seq_max = N_CTX, N_BATCH, N_UBATCH, n_seq
        params.n_threads = params.n_threads_batch = threads
        params.kv_unified = True  # one cache for every sequence: the label sequences share the prompt's cells
        if device != "metal":
            params.offload_kqv = params.op_offload = False
        self.ctx = ll.llama_init_from_model(self.model, params)
        assert self.ctx, "llama.cpp could not create a context"
        self.memory = ll.llama_get_memory(self.ctx)
        self.n_vocab = ll.llama_vocab_n_tokens(ll.llama_model_get_vocab(self.model))
        self.batch = ll.llama_batch_init(N_BATCH, 0, 1)

    def __del__(self):
        # Freed before the process exits: Metal aborts if its buffers outlive the device.
        self.ll.llama_batch_free(self.batch)
        self.ll.llama_free(self.ctx)
        self.ll.llama_model_free(self.model)

    def decode(self, tokens: list[tuple[int, int, int, bool]]) -> np.ndarray:
        """Run (token, position, sequence, wants logits) entries; the logits of those that want them."""
        rows = []
        for start in range(0, len(tokens), N_BATCH):
            part = tokens[start : start + N_BATCH]
            self.batch.n_tokens = len(part)
            for i, (token, pos, seq, logits) in enumerate(part):
                self.batch.token[i], self.batch.pos[i], self.batch.logits[i] = token, pos, logits
                self.batch.n_seq_id[i], self.batch.seq_id[i][0] = 1, seq
            status = self.ll.llama_decode(self.ctx, self.batch)
            assert status == 0, f"llama_decode returned {status}"
            # The rows that want logits are stored contiguously, in batch order.
            n_out = sum(entry[3] for entry in part)
            if n_out:
                logits = self.ll.llama_get_logits(self.ctx)
                rows.append(np.ctypeslib.as_array(logits, shape=(n_out, self.n_vocab)).copy())
        return np.concatenate(rows) if rows else np.empty((0, self.n_vocab), dtype=np.float32)

    def prompt(self, ids: list[int], seq: int, start: int = 0) -> np.ndarray:
        """Append ``ids`` to a sequence that holds ``start`` tokens; the logits after the last one."""
        n = len(ids)
        return self.decode([(t, start + i, seq, i == n - 1) for i, t in enumerate(ids)])[0]

    def copy(self, src: int, dst: int) -> None:
        self.ll.llama_memory_seq_cp(self.memory, src, dst, -1, -1)

    def clear(self, seq: int) -> None:
        self.ll.llama_memory_seq_rm(self.memory, seq, -1, -1)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    top = logits.max(axis=-1, keepdims=True)
    return logits - top - np.log(np.exp(logits - top).sum(axis=-1, keepdims=True))


def _logprobs(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """log p(targets[i]) under row i of ``logits``, without a log-softmax over the whole vocabulary."""
    top = logits.max(axis=-1)
    norm = top + np.log(np.exp(logits - top[:, None]).sum(axis=-1))
    return logits[np.arange(len(targets)), targets] - norm


class SlotScorer:
    """Next-token distribution over the slot tokens at the end of a request, as ``decision.SlotScorer``."""

    def __init__(self, gguf: str, device: str, repo: str, revision: str):
        self.chat = Chat(repo, revision)
        ids = [self.chat.encode(label) for label in SLOT_LABELS]
        assert all(len(i) == 1 for i in ids), "slot labels must be single tokens"
        self.slot_ids = np.array([i[0] for i in ids])
        self.runtime = Runtime(gguf, device)

    def score(self, chat: list[dict[str, str]], n_options: int) -> np.ndarray:
        self.runtime.clear(0)
        logits = self.runtime.prompt(self.chat.prompt_ids(chat), 0)[self.slot_ids[:n_options]]
        return np.exp(_log_softmax(logits))


class LabelScorer:
    """Probability of each label as the complete answer of the assistant turn, as ``prompted.LabelScorer``."""

    PREFIX, TEXT = 0, 1  # sequences; the labels of a chunk use 2 to 2 + LABEL_CHUNK - 1

    def __init__(self, gguf: str, device: str, repo: str, revision: str):
        self.chat = Chat(repo, revision)
        self.end_of_turn = self.chat.token_id("<|im_end|>")
        self.runtime = Runtime(gguf, device, n_seq=2 + LABEL_CHUNK)
        self._label_ids: dict[tuple[str, ...], list[list[int]]] = {}
        self._prefix: list[int] = []

    def prompt_ids(self, message: str) -> list[int]:
        return self.chat.prompt_ids([{"role": "user", "content": message}])

    def label_ids(self, labels: list[str]) -> list[list[int]]:
        key = tuple(labels)
        if key not in self._label_ids:
            self._label_ids[key] = [self.chat.encode(label) + [self.end_of_turn] for label in labels]
        return self._label_ids[key]

    def share_prefix(self, *messages: str) -> int:
        ids = [self.prompt_ids(m) for m in messages]
        n = common_prefix_length(ids)
        self.runtime.clear(self.PREFIX)
        self._prefix = ids[0][:n]
        if n:
            self.runtime.prompt(self._prefix, self.PREFIX)
        return n

    def score(self, message: str, labels: list[str]) -> np.ndarray:
        prompt, label_ids = self.prompt_ids(message), self.label_ids(labels)
        runtime, n = self.runtime, len(self._prefix)
        runtime.clear(self.TEXT)
        if 0 < n < len(prompt) and prompt[:n] == self._prefix:
            runtime.copy(self.PREFIX, self.TEXT)
            last = runtime.prompt(prompt[n:], self.TEXT, start=n)
        else:
            last = runtime.prompt(prompt, self.TEXT)
        first = np.array([ids[0] for ids in label_ids])
        logprobs = _logprobs(np.broadcast_to(last, (1, len(last))), first[:1]).astype(np.float64)
        logprobs = logprobs[0] - last[first[0]] + last[first]  # same normalizer for every first token
        for chunk_start in range(0, len(label_ids), LABEL_CHUNK):
            chunk = label_ids[chunk_start : chunk_start + LABEL_CHUNK]
            # Every label of a chunk is right-padded to the same width, as in MLX: llama.cpp splits a batch
            # of recurrent sequences into equal-length pieces, so unequal labels would cost several passes.
            # Padding comes after the real tokens and wants no logits, so it never changes what is read.
            width = max(len(ids) for ids in chunk) - 1
            tokens, reads = [], []
            for j, ids in enumerate(chunk):
                seq = 2 + j
                runtime.copy(self.TEXT, seq)
                padded = ids[:-1] + [ids[-1]] * (width - len(ids) + 1)
                tokens += [(t, len(prompt) + k, seq, k < len(ids) - 1) for k, t in enumerate(padded)]
                reads.append((chunk_start + j, ids[1:]))
            if not tokens:
                continue
            logp = _logprobs(runtime.decode(tokens), np.concatenate([t for _, t in reads]))
            row = 0
            for label, targets in reads:
                logprobs[label] += float(logp[row : row + len(targets)].sum())
                row += len(targets)
            for j in range(len(chunk)):
                runtime.clear(2 + j)
        probs = np.exp(logprobs - logprobs.max())
        return probs / probs.sum()


@dataclass(frozen=True)
class GgufPromptedLLM(PromptedLLM):
    """``repo`` and ``revision`` give the tokenizer and chat template; ``gguf`` the weights."""

    gguf: str = ""
    device: str = "metal"

    def identity(self) -> dict:
        return super().identity() | {
            "runtime": "llama.cpp",
            "gguf": Path(self.gguf).name,
            "device": self.device,
        }

    def load(self, out_dir: Path) -> PromptedPredictor:
        spec = json.loads((out_dir / PROMPT_SPEC).read_text())
        return PromptedPredictor(LabelScorer(self.gguf, self.device, self.repo, self.revision), **spec)

    def disk_mb(self, out_dir: Path) -> float:
        return round(Path(self.gguf).stat().st_size / 2**20, 1)


@dataclass(frozen=True)
class GgufDecisionModel(DecisionModel):
    """``repo`` and ``revision`` give the tokenizer and chat template; ``gguf`` the weights."""

    gguf: str = ""
    device: str = "metal"

    def identity(self) -> dict:
        return super().identity() | {"runtime": "llama.cpp", "gguf": Path(self.gguf).name}

    def load(self, out_dir: Path) -> DecisionPredictor:
        spec = json.loads((out_dir / LABELS_FILE).read_text())
        scorer = SlotScorer(self.gguf, self.device, self.repo, self.revision)
        return DecisionPredictor(scorer, spec["task"], spec["labels"])

    def disk_mb(self, out_dir: Path) -> float:
        return round(Path(self.gguf).stat().st_size / 2**20, 1)
