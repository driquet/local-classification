"""Toy tasks written in the frozen-split format, so data, prompts and runs are tested without downloads."""

import pandas as pd
import pytest

from lc.data import CLINC150, OUT_OF_SCOPE, class_slice, slice_name, write_task

INTENTS = [f"intent_{i:03d}" for i in range(150)]
STANCES = ["against", "for", "neutral"]
SENTIMENTS = ["positive", "negative", "neutral"]


def toy_splits(labels: list[str], per_label: int = 20) -> dict[str, pd.DataFrame]:
    """``per_label`` examples per label in every split, with distinct texts and ids."""
    rows = [(f"{label} example {i}.", label) for i in range(per_label) for label in labels]
    return {
        name: pd.DataFrame(
            {
                "id": [f"{name}-{i:05d}" for i in range(len(rows))],
                "text": [text for text, _ in rows],
                "label": [label for _, label in rows],
            }
        )
        for name in ("train", "validation", "test")
    }


@pytest.fixture
def data_dir(tmp_path):
    """A data directory with toy VAST, toy DynaSent, toy CLINC150 and one 10-intent slice of chain 0."""
    write_task(tmp_path / "vast", toy_splits(STANCES), STANCES, {})
    write_task(tmp_path / "dynasent", toy_splits(SENTIMENTS), SENTIMENTS, {})
    labels = sorted([*INTENTS, OUT_OF_SCOPE])
    splits = toy_splits(labels, per_label=4)
    write_task(tmp_path / CLINC150, splits, labels, {"n_classes": 151})
    in_scope = {name: df[df["label"] != OUT_OF_SCOPE] for name, df in splits.items()}
    sliced, slice_labels = class_slice(in_scope, INTENTS, 10, 0)
    write_task(tmp_path / slice_name(10, 0), sliced, slice_labels, {"n_classes": 10, "chain": 0})
    return tmp_path
