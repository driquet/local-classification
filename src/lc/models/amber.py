"""Amber: decision models built on an encoder, from 32M to 400M parameters (Decision Index entrants).

Each Amber checkpoint is an Ettin reranker trained to read a request (instructions, the options, a state) in
one encoder pass and return a probability per option, as a decision model does. Like OneJev, it needs no
labels from the task; unlike a fine-tuned encoder, its option list is part of the input, so the labels can
change without training.

The model repository ships its own Decision Index engine (``amber_uni_engine.py`` and
``amber_uni_core.py``), run here as is, with each checkpoint's own defaults (bf16, its number of option
orders). The engine subclasses the Decision Index kit's ``Engine`` and raises its ``Unsupported``: two names,
stubbed here so that the kit need not be installed. Requests are worded as the Decision Index words them,
with ``decision.request``, the same wording OneJev gets.

Precisions: the engine runs bf16 (its default, as on the Decision Index) or fp32. ``AmberInt8`` serves the
same weights in int8 with ONNX Runtime on the CPU, as the fine-tuned encoders' int8: the encoder and the
option head are exported for one unpadded sequence, the request is assembled here as the engine assembles
it, and prediction imports neither torch nor transformers.

Training data: the Amber cards list CLINC150's training split among their sources (the Decision Index allows
training splits, never test rows), so on CLINC150 Amber is not a zero-shot model. DynaSent is not in the
Decision Index suite.
"""

import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lc.measure import disk_mb, in_ram
from lc.models.decision import LABELS_FILE, request

DEVICE = "mps"


@dataclass(frozen=True)
class AmberDecisionModel:
    """The repository's engine, ``dtype`` ``bf16`` or ``fp32``, on ``mps`` or ``cpu`` (``threads``)."""

    repo: str
    revision: str
    device: str = DEVICE
    threads: int | None = None
    dtype: str = "bf16"

    def identity(self) -> dict:
        return {"approach": "amber-decision-model", **asdict(self)}

    def fit(self, task: str, train: pd.DataFrame, labels: list[str], seed: int, out_dir: Path) -> int:
        """Nothing to train: record the task and its labels."""
        (out_dir / LABELS_FILE).write_text(json.dumps({"task": task, "labels": labels}))
        return 0

    def load(self, out_dir: Path) -> "AmberPredictor":
        spec = json.loads((out_dir / LABELS_FILE).read_text())
        model = engine(self.repo, self.revision, self.device, self.dtype, self.threads)
        return AmberPredictor(model, **spec)

    def disk_mb(self, out_dir: Path) -> float:
        return disk_mb(_snapshot(self.repo, self.revision))


class AmberPredictor:
    def __init__(self, engine, task: str, labels: list[str]):
        self.engine, self.task, self.labels = engine, task, labels

    def predict(self, text: str) -> np.ndarray:
        instructions, options, order = request(self.task, text, self.labels)
        question = {"type": "choice", "instructions": instructions, "criteria": dict(options)}
        answer = self.engine({}, {"q": question})[0]["answers"]["q"]["probabilities"]
        probs = np.array([answer[key] for key, _ in options])
        return probs[[order.index(label) for label in self.labels]]


def engine(repo: str, revision: str, device: str, dtype: str = "bf16", threads: int | None = None):
    """The repository's own engine, loaded from its snapshot."""
    import torch

    if threads:
        torch.set_num_threads(threads)
    path = _snapshot(repo, revision)
    _stub_decision_index_kit()
    sys.path.insert(0, str(path))
    from amber_uni_engine import UniEngine

    model = UniEngine(model_dir=str(path), device=device, dtype=dtype)
    if device == "cpu":
        in_ram(model.model)
    return model


INT8_FILE = "amber.int8.onnx"
TOKENS_FILE = "tokens.json"


@dataclass(frozen=True)
class AmberInt8:
    """The same weights in int8, ONNX Runtime on ``threads`` CPU threads. ``exports`` caches the export."""

    repo: str
    revision: str
    exports: str
    threads: int = 1

    def identity(self) -> dict:
        return {
            "approach": "amber-decision-model-int8",
            "repo": self.repo,
            "revision": self.revision,
            "threads": self.threads,
        }

    def fit(self, task: str, train: pd.DataFrame, labels: list[str], seed: int, out_dir: Path) -> int:
        exported = Path(self.exports) / f"{self.repo.split('/')[1]}-{self.revision[:8]}"
        if not (exported / INT8_FILE).exists():
            export_int8(_snapshot(self.repo, self.revision), exported)
        (out_dir / LABELS_FILE).write_text(json.dumps({"task": task, "labels": labels}))
        (out_dir / "export").write_text(str(exported))
        return 0

    def load(self, out_dir: Path) -> "AmberPredictor":
        spec = json.loads((out_dir / LABELS_FILE).read_text())
        exported = Path((out_dir / "export").read_text())
        return AmberPredictor(OnnxEngine(exported, _snapshot(self.repo, self.revision), self.threads), **spec)

    def disk_mb(self, out_dir: Path) -> float:
        exported = Path((out_dir / "export").read_text())
        return disk_mb(exported) + disk_mb(_snapshot(self.repo, self.revision) / "encoder")


def export_int8(snapshot: Path, out_dir: Path) -> None:
    """The encoder and option head for one unpadded sequence: (input_ids, marker, start, end) -> scores."""
    import onnx
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoTokenizer

    class Scores(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, marker, start, end):
            h = self.model.encoder(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
            h = h.last_hidden_state[0]
            cs = torch.nn.functional.pad(h.cumsum(0), (0, 0, 1, 0))
            mean = (cs[end] - cs[start]) / (end - start).clamp(min=1).unsqueeze(-1).float()
            first = h[0].unsqueeze(0).expand(marker.shape[0], -1)
            return self.model.head(torch.cat([h[marker], mean, first], dim=-1)).squeeze(-1)

    model = engine(str(snapshot), None, "cpu", "fp32").model.float().eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    float_path = out_dir / "amber.onnx"
    example = (
        torch.ones((1, 32), dtype=torch.long),
        torch.tensor([10, 20]),
        torch.tensor([11, 21]),
        torch.tensor([13, 23]),
    )
    length, options = torch.export.Dim("length", min=4, max=8000), torch.export.Dim("options", min=1, max=512)
    torch.onnx.export(
        Scores(model),
        example,
        float_path,
        input_names=["input_ids", "marker", "start", "end"],
        output_names=["scores"],
        dynamic_shapes=({1: length}, {0: options}, {0: options}, {0: options}),
        dynamo=True,
        external_data=False,
    )
    exported = onnx.load(float_path)
    del exported.graph.value_info[:]
    onnx.save(exported, float_path)
    quantize_dynamic(float_path, out_dir / INT8_FILE, per_channel=True, weight_type=QuantType.QInt8)
    float_path.unlink()
    tok = AutoTokenizer.from_pretrained(snapshot / "encoder")
    special = {"cls": tok.cls_token_id, "sep": tok.sep_token_id, "mask": tok.mask_token_id}
    orders = json.loads((snapshot / "uni_config.json").read_text()).get("orders", 1)
    (out_dir / TOKENS_FILE).write_text(json.dumps({**special, "orders": orders}))


class OnnxEngine:
    """The engine's ``__call__`` for one choice question and an empty state, on the int8 export."""

    def __init__(self, exported: Path, snapshot: Path, threads: int):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(snapshot / "encoder" / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self.special = json.loads((exported / TOKENS_FILE).read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        self.session = ort.InferenceSession(exported / INT8_FILE, options, providers=["CPUExecutionProvider"])

    def ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def __call__(self, state, questions):
        ((name, question),) = questions.items()
        keys = list(question["criteria"])
        instructions = self.ids(question["instructions"])
        option_ids = [self.ids(question["criteria"][k] or k) for k in keys]
        cls, sep, mask = self.special["cls"], self.special["sep"], self.special["mask"]
        sums = np.zeros(len(keys))
        orders = option_orders(len(keys), self.special["orders"])
        for perm in orders:
            # As amber_uni_core.assemble: [CLS] instructions [SEP] state (empty) [SEP], then [MASK] option ...
            ids = [cls, *instructions, sep, sep]
            marker, start, end = [], [], []
            for j in perm:
                marker.append(len(ids))
                ids.append(mask)
                start.append(len(ids))
                ids.extend(option_ids[j])
                end.append(len(ids))
            ids.append(sep)
            feeds = {
                "input_ids": np.array([ids], dtype=np.int64),
                "marker": np.array(marker, dtype=np.int64),
                "start": np.array(start, dtype=np.int64),
                "end": np.array(end, dtype=np.int64),
            }
            scores = self.session.run(None, feeds)[0].astype(np.float64)
            probs = np.exp(scores - scores.max())
            sums[perm] += probs / probs.sum()
        dist = dict(zip(keys, sums / len(orders), strict=True))
        return {"answers": {name: {"type": "choice", "probabilities": dist}}}, None


def option_orders(n: int, k: int) -> list[list[int]]:
    """As the engine's: the first ``k`` distinct of given, reversed, rotated by half, rotated and reversed."""
    given = list(range(n))
    rotated = given[n // 2 :] + given[: n // 2]
    out = []
    for p in (given, given[::-1], rotated, rotated[::-1])[:k]:
        if p not in out:
            out.append(p)
    return out


def _snapshot(repo: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(repo) if Path(repo).is_dir() else Path(snapshot_download(repo, revision=revision))


def _stub_decision_index_kit() -> None:
    """``decision_index.engines.base``: the ``Engine`` base class and the ``Unsupported`` error."""
    if "decision_index.engines.base" in sys.modules:
        return

    class Engine:
        def __init__(self, **options):
            self.options = options

    class Unsupported(Exception):
        pass

    base = types.ModuleType("decision_index.engines.base")
    base.Engine, base.Unsupported = Engine, Unsupported
    for name in ("decision_index", "decision_index.engines"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["decision_index.engines.base"] = base
