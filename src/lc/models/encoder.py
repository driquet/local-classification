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
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as transformers_logging

from lc.measure import disk_mb

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

    def identity(self) -> dict:
        return {"approach": "fine-tuned-encoder", **asdict(self)}

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
        texts = train["text"].tolist()
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
                encoded = tokenizer(
                    [texts[i] for i in ids],
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(DEVICE)
                loss = model(**encoded, labels=targets[ids].to(DEVICE)).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
                step += 1
                if step % 200 == 0:
                    print(f"    step {step}/{total_steps}, loss {loss.item():.3f}", flush=True)
                if step == total_steps:
                    break
        torch.mps.synchronize()
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        (out_dir / "labels.json").write_text(json.dumps(labels))
        return len(train)

    def load(self, out_dir: Path) -> "EncoderPredictor":
        return EncoderPredictor(out_dir, self.max_length)

    def disk_mb(self, out_dir: Path) -> float:
        return disk_mb(out_dir)


class EncoderPredictor:
    def __init__(self, model_dir: Path, max_length: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE).eval()
        self.max_length = max_length

    @torch.inference_mode()
    def predict(self, text: str) -> np.ndarray:
        encoded = self.tokenizer([text], truncation=True, max_length=self.max_length, return_tensors="pt")
        logits = self.model(**encoded.to(DEVICE)).logits[0].float()
        return torch.softmax(logits, dim=-1).cpu().numpy()
