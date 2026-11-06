"""A pretrained encoder left as it is, with a logistic regression on its embeddings.

Each text becomes the mean of its token vectors, normalized to length one, which is how sentence-embedding
models are used. Only the regression trains, in seconds, so this is the cheapest way to use an encoder, and
what fine-tuning has to beat.
"""

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

from lc.measure import disk_mb, in_ram

FIT_DEVICE = "mps"
HEAD_FILE = "head.pkl"


def embed(model, tokenizer, texts: list[str], max_length: int, device: str, batch_size: int) -> np.ndarray:
    """Embeddings in the order of ``texts``; batches group similar lengths, which saves padding."""
    order = np.argsort([len(text) for text in texts])
    vectors = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        ids = order[start : start + batch_size]
        encoded = tokenizer(
            [texts[i] for i in ids], truncation=True, max_length=max_length, padding=True, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            tokens = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(tokens.dtype)
        mean = (tokens * mask).sum(1) / mask.sum(1)
        vectors[ids] = torch.nn.functional.normalize(mean, dim=-1).cpu().numpy()
    return vectors


@dataclass(frozen=True)
class FrozenEncoder:
    repo: str
    revision: str
    max_length: int = 512
    batch_size: int = 32
    c: float = 10.0  # inverse regularization: unit vectors need more than scikit-learn's default of 1
    max_iter: int = 2000
    device: str = FIT_DEVICE  # where it serves; training always runs on the GPU
    threads: int | None = None

    def identity(self) -> dict:
        return {"approach": "frozen-encoder-logistic-regression", **asdict(self)}

    def fit(self, task: str, train: pd.DataFrame, labels: list[str], seed: int, out_dir: Path) -> int:
        tokenizer = AutoTokenizer.from_pretrained(self.repo, revision=self.revision)
        model = AutoModel.from_pretrained(self.repo, revision=self.revision).to(FIT_DEVICE).eval()
        texts = train["text"].tolist()
        vectors = embed(model, tokenizer, texts, self.max_length, FIT_DEVICE, self.batch_size)
        head = LogisticRegression(C=self.c, max_iter=self.max_iter, random_state=seed)
        head.fit(vectors, train["label"])
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        (out_dir / HEAD_FILE).write_bytes(pickle.dumps((head, labels)))
        return len(train)

    def load(self, out_dir: Path) -> "FrozenPredictor":
        if self.threads:
            torch.set_num_threads(self.threads)
        return FrozenPredictor(out_dir, self.max_length, self.device)

    def disk_mb(self, out_dir: Path) -> float:
        return disk_mb(out_dir)


class FrozenPredictor:
    def __init__(self, model_dir: Path, max_length: int, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(device).eval()
        if device == "cpu":
            in_ram(self.model)
        self.head, labels = pickle.loads((model_dir / HEAD_FILE).read_bytes())
        # predict_proba follows head.classes_ (sorted); reorder its columns to the task's label order.
        self.order = [list(self.head.classes_).index(label) for label in labels]
        self.max_length, self.device = max_length, device

    def predict(self, text: str) -> np.ndarray:
        vector = embed(self.model, self.tokenizer, [text], self.max_length, self.device, 1)
        return self.head.predict_proba(vector)[0][self.order]
