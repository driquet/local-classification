"""A fine-tuned checkpoint served in int8 by ONNX Runtime, on the CPU.

Only weights of the linear layers and embeddings become 8-bit integers; activations are quantized on the
fly, per text (dynamic quantization). Prediction needs neither torch nor transformers, which is why this
module imports them only inside the export: its memory is what serving the model takes without them.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from lc.measure import disk_mb

MODEL_FILE = "model.int8.onnx"
TOKENIZER_FILE = "tokenizer.json"
LABELS_FILE = "labels.json"


def export_int8(checkpoint: Path, out_dir: Path) -> None:
    import onnx
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification

    class Logits(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    model = Logits(AutoModelForSequenceClassification.from_pretrained(checkpoint)).eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    float_path = out_dir / "model.onnx"
    example = (torch.ones((2, 16), dtype=torch.long), torch.ones((2, 16), dtype=torch.long))
    length = torch.export.Dim("length", min=1, max=8000)
    batch = torch.export.Dim("batch", min=1, max=512)
    torch.onnx.export(
        model,
        example,
        float_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes=({0: batch, 1: length}, {0: batch, 1: length}),
        dynamo=True,
        external_data=False,
    )
    # The exporter's shape annotations of the head trip the quantizer's shape inference, which redoes them.
    exported = onnx.load(float_path)
    del exported.graph.value_info[:]
    onnx.save(exported, float_path)
    # One scale per channel: on Ettin-17M it halves the error of the logits, for the same size.
    quantize_dynamic(float_path, out_dir / MODEL_FILE, per_channel=True, weight_type=QuantType.QInt8)
    float_path.unlink()


@dataclass(frozen=True)
class OnnxEncoder:
    """``base`` is the identity of the fine-tuned encoder, as JSON: importing it here would load torch."""

    base: str
    checkpoints: str
    max_length: int
    threads: int

    def identity(self) -> dict:
        return {
            "approach": "onnx-int8-encoder",
            "base": json.loads(self.base),
            "threads": self.threads,
        }

    def fit(self, task: str, train, labels: list[str], seed: int, out_dir: Path) -> int:
        checkpoint = Path(self.checkpoints) / task / f"seed-{seed}"
        exported = checkpoint.with_name(f"{checkpoint.name}-int8")
        if not (exported / MODEL_FILE).exists():
            export_int8(checkpoint, exported)
        shutil.copy(exported / MODEL_FILE, out_dir)
        shutil.copy(checkpoint / TOKENIZER_FILE, out_dir)
        shutil.copy(checkpoint / LABELS_FILE, out_dir)
        return len(train)

    def load(self, out_dir: Path) -> "OnnxPredictor":
        return OnnxPredictor(out_dir, self.max_length, self.threads)

    def disk_mb(self, out_dir: Path) -> float:
        return disk_mb(out_dir)


class OnnxPredictor:
    def __init__(self, model_dir: Path, max_length: int, threads: int):
        self.tokenizer = Tokenizer.from_file(str(model_dir / TOKENIZER_FILE))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length)
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        self.session = ort.InferenceSession(
            model_dir / MODEL_FILE, options, providers=["CPUExecutionProvider"]
        )

    def predict(self, text: str) -> np.ndarray:
        encoded = self.tokenizer.encode(text)
        logits = self.session.run(
            None,
            {
                "input_ids": np.array([encoded.ids], dtype=np.int64),
                "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
            },
        )[0][0].astype(np.float64)
        probs = np.exp(logits - logits.max())
        return probs / probs.sum()
