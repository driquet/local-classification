"""A pretrained encoder fine-tuned with a classification head.

The encoder reads the whole text at once, and a linear head on top of its pooled representation gives
one logit per label. Every weight is fine-tuned on the task's training set.

The recipe follows common practice for BERT-style models and is the same for every task:
3 epochs, batch size 16, AdamW with weight decay 0.01, a linear schedule with 10% warmup, and a learning
rate of 5e-5, the value the ModernBERT paper reports for its base-size model on MNLI (Ettin reuses
ModernBERT's architecture and training recipe). Two details matter more than they seem:

- Gradients are clipped at a norm of 1.0, as the Hugging Face ``Trainer`` does by default. Without it, one
  unlucky batch early in training can push the model into predicting a single class for good.
- Training runs at least ``min_steps`` steps. Three epochs over 10 examples per class are only a few dozen
  steps, too few for the head to move away from its random initialization; the floor gives small training
  sets a fair chance without changing anything for large ones.
"""

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as transformers_logging

from lc.measure import disk_mb, in_ram

DEVICE = "mps"
# Loading a pretrained encoder with a new classification head reports the head's weights as missing,
# which is expected: they are what fine-tuning learns.
transformers_logging.set_verbosity_error()


@dataclass(frozen=True)
class FineTunedEncoder:
    repo: str
    revision: str
    learning_rate: float = 5e-5
    epochs: int = 3
    batch_size: int = 16
    min_steps: int = 200
    warmup: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_length: int = 512
    # Recompute each layer's activations in the backward pass instead of keeping them: same updates, about a
    # third more time, a fraction of the memory. Ettin-400M needs it to train on VAST in 24 GB.
    gradient_checkpointing: bool = False
    # Distillation: learn a teacher's probabilities (the training set's ``probs`` column) instead of its
    # labels, with the KL divergence between both, softened by ``temperature`` and scaled by its square
    # (Hinton et al., 2015). The teacher returns probabilities, not logits: softening raises them to 1/T.
    soft: bool = False
    temperature: float = 1.0
    # Split each batch into this many parts, one forward and backward pass each, and update once: the same
    # update as the whole batch, for a fraction of the activation memory. 512-token abstracts need it.
    accumulation: int = 1

    def identity(self) -> dict:
        # Fields added after the first runs are left out at their default, so those runs keep their ids.
        late = {"gradient_checkpointing": False, "soft": False, "temperature": 1.0, "accumulation": 1}
        fields = {k: v for k, v in asdict(self).items() if k not in late or v != late[k]}
        return {"approach": "fine-tuned-encoder", **fields}

    def fit(self, task: str, train: pd.DataFrame, labels: list[str], seed: int, out_dir: Path) -> int:
        torch.manual_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(self.repo, revision=self.revision)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.repo,
            revision=self.revision,
            num_labels=len(labels),
            id2label=dict(enumerate(labels)),
            label2id={label: i for i, label in enumerate(labels)},
        ).to(DEVICE)
        if self.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        texts = train["text"].tolist()
        if self.soft:
            teacher = torch.tensor(np.stack(train["probs"].to_numpy()), dtype=torch.float32)
            teacher = teacher ** (1 / self.temperature)
            targets = teacher / teacher.sum(dim=1, keepdim=True)
        else:
            targets = torch.tensor([labels.index(label) for label in train["label"]])
        total_steps = max(self.epochs * math.ceil(len(texts) / self.batch_size), self.min_steps)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        schedule = get_linear_schedule_with_warmup(optimizer, int(self.warmup * total_steps), total_steps)
        rng = np.random.default_rng(seed)

        model.train()
        step = 0
        while step < total_steps:
            order = rng.permutation(len(texts))
            for start in range(0, len(order), self.batch_size):
                ids = order[start : start + self.batch_size]
                size = math.ceil(len(ids) / self.accumulation)
                for part in (ids[i : i + size] for i in range(0, len(ids), size)):
                    loss = self._loss(model, tokenizer, [texts[i] for i in part], targets[part])
                    (loss * len(part) / len(ids)).backward()  # weighted: the mean over the whole batch
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
                step += 1
                # Batches padded to their own length leave odd-sized blocks in the MPS cache, which grows
                # until the machine swaps on long texts (VAST): release it now and then.
                if step % 50 == 0:
                    torch.mps.empty_cache()
                if step % 200 == 0:
                    print(f"    step {step}/{total_steps}, loss {loss.item():.3f}", flush=True)
                if step == total_steps:
                    break
        torch.mps.synchronize()
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        (out_dir / "labels.json").write_text(json.dumps(labels))
        return len(train)

    def _loss(self, model, tokenizer, texts: list[str], targets: torch.Tensor) -> torch.Tensor:
        """Mean loss over ``texts``: cross-entropy on labels, or softened KL on a teacher's probabilities."""
        encoded = tokenizer(
            texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt"
        ).to(DEVICE)
        if not self.soft:
            return model(**encoded, labels=targets.to(DEVICE)).loss
        t = self.temperature
        log_student = torch.log_softmax(model(**encoded).logits / t, dim=-1)
        return torch.nn.functional.kl_div(log_student, targets.to(DEVICE), reduction="batchmean") * (t * t)

    def load(self, out_dir: Path) -> "EncoderPredictor":
        return EncoderPredictor(out_dir, self.max_length)

    def disk_mb(self, out_dir: Path) -> float:
        return disk_mb(out_dir)


@dataclass(frozen=True)
class ServedEncoder:
    """A fine-tuned checkpoint served another way: other device, thread count or precision.

    ``fit`` trains nothing. It takes the checkpoint of the same task and seed under ``checkpoints``, so
    every variant of a model shares its weights and only the serving differs.
    """

    base: FineTunedEncoder
    checkpoints: str
    device: str = DEVICE
    threads: int | None = None
    half: bool = False

    def identity(self) -> dict:
        return {
            "approach": "served-encoder",
            "base": self.base.identity(),
            "device": self.device,
            "threads": self.threads,
            "half": self.half,
        }

    def fit(self, task: str, train: pd.DataFrame, labels: list[str], seed: int, out_dir: Path) -> int:
        shutil.copytree(Path(self.checkpoints) / task / f"seed-{seed}", out_dir, dirs_exist_ok=True)
        if self.half:
            AutoModelForSequenceClassification.from_pretrained(out_dir).half().save_pretrained(out_dir)
        return len(train)

    def load(self, out_dir: Path) -> "EncoderPredictor":
        if self.threads:
            torch.set_num_threads(self.threads)
        return EncoderPredictor(out_dir, self.base.max_length, self.device, self.half)

    def disk_mb(self, out_dir: Path) -> float:
        return disk_mb(out_dir)


class EncoderPredictor:
    def __init__(self, model_dir: Path, max_length: int, device: str = DEVICE, half: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        options = {"dtype": torch.float16} if half else {}
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, **options)
        self.model = self.model.to(device).eval()
        if device == "cpu":
            in_ram(self.model)
        self.max_length, self.device = max_length, device

    @torch.inference_mode()
    def predict(self, text: str) -> np.ndarray:
        encoded = self.tokenizer([text], truncation=True, max_length=self.max_length, return_tensors="pt")
        logits = self.model(**encoded.to(self.device)).logits[0].float()
        return torch.softmax(logits, dim=-1).cpu().numpy()
