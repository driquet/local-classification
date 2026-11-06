"""The four approaches, behind one interface.

A classifier object holds configuration only, so a run can send it to a fresh process. ``fit`` writes
whatever prediction needs to a directory: trained weights for TF-IDF and the encoder, the rendered prompt
for the prompted LLM, the label list for the decision model. ``load`` reads that directory back in the
prediction process and returns a predictor, which answers one text at a time with one probability per
label, in the task's label order.
"""

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


class Predictor(Protocol):
    def predict(self, text: str) -> np.ndarray: ...


class Classifier(Protocol):
    def identity(self) -> dict:
        """Everything that defines the approach (model, revision, settings), hashed into run ids."""
        ...

    def fit(self, task: str, train: pd.DataFrame, labels: list[str], seed: int, out_dir: Path) -> int:
        """Prepare prediction in ``out_dir``; returns the number of labeled examples used."""
        ...

    def load(self, out_dir: Path) -> Predictor: ...

    def disk_mb(self, out_dir: Path) -> float:
        """What deploying the approach puts on disk."""
        ...
