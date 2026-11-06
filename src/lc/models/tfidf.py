"""TF-IDF and logistic regression: the floor every other approach has to beat.

Each text becomes a sparse vector of word unigrams and bigrams plus character 2- to 5-grams (within word
boundaries, which tolerates typos and inflections), weighted by TF-IDF with sublinear term frequency.
A multinomial logistic regression classifies it. No GPU and no pretrained weights; training takes seconds.
"""

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline, make_union

from lc.measure import disk_mb

MODEL_FILE = "model.pkl"


@dataclass(frozen=True)
class TfidfLogisticRegression:
    word_ngrams: tuple[int, int] = (1, 2)
    char_ngrams: tuple[int, int] = (2, 5)
    max_iter: int = 2000

    def identity(self) -> dict:
        return {"approach": "tfidf-logistic-regression", **asdict(self)}

    def fit(self, task: str, train: pd.DataFrame, labels: list[str], seed: int, out_dir: Path) -> int:
        model = make_pipeline(
            make_union(
                TfidfVectorizer(ngram_range=self.word_ngrams, sublinear_tf=True),
                TfidfVectorizer(analyzer="char_wb", ngram_range=self.char_ngrams, sublinear_tf=True),
            ),
            LogisticRegression(max_iter=self.max_iter, random_state=seed),
        )
        model.fit(train["text"], train["label"])
        (out_dir / MODEL_FILE).write_bytes(pickle.dumps((model, labels)))
        return len(train)

    def load(self, out_dir: Path) -> "TfidfPredictor":
        model, labels = pickle.loads((out_dir / MODEL_FILE).read_bytes())
        return TfidfPredictor(model, labels)

    def disk_mb(self, out_dir: Path) -> float:
        return disk_mb(out_dir)


class TfidfPredictor:
    def __init__(self, model: Pipeline, labels: list[str]):
        self.model = model
        # predict_proba follows model.classes_ (sorted); reorder its columns to the task's label order.
        self.order = [list(model.classes_).index(label) for label in labels]

    def predict(self, text: str) -> np.ndarray:
        return self.model.predict_proba([text])[0][self.order]
