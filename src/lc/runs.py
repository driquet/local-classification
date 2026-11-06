"""Runs: one approach on one task, from training to a row in the results table.

A run fits the classifier in a fresh process, predicts the test set one text at a time in another fresh
process, writes its predictions to ``<runs_dir>/<task>/<run id>/``, and appends one row to the results
table. The run id hashes everything that defines the run (approach settings, task, data version, training
subset, seed), so a run already in the table is skipped: an interrupted experiment resumes where it stopped.
"""

import csv
import hashlib
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from lc import data
from lc.measure import in_fresh_process
from lc.metrics import per_class_f1, scores
from lc.models import Classifier

COLUMNS = [
    "run_id",
    "approach",
    "task",
    "split",
    "n_classes",
    "chain",
    "per_class",
    "seed",
    "data_version",
    "train_n",
    "fit_s",
    "train_peak_mb",
    "predict_s",
    "peak_mb",
    "disk_mb",
    "n",
    "macro_f1",
    "macro_f1_low",
    "macro_f1_high",
    "accuracy",
    "accuracy_low",
    "accuracy_high",
    "per_class_f1",
    "latency_p50_ms",
    "latency_p95_ms",
    "evaluated_at",
]


@dataclass(frozen=True)
class Run:
    """One approach on one task. ``per_class`` caps the training examples per class (``None``: all)."""

    approach: str
    task: str
    per_class: int | None = None
    seed: int = 0
    split: str = "test"


def run_id(run: Run, classifier: Classifier, limit: int | None = None, data_dir: Path = data.DATA_DIR) -> str:
    identity = {
        **asdict(run),
        "classifier": classifier.identity(),
        "data_version": data.data_version(run.task, data_dir),
        "limit": limit,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]


def completed(results_path: Path) -> set[str]:
    if not Path(results_path).exists():
        return set()
    with Path(results_path).open() as f:
        return {row["run_id"] for row in csv.DictReader(f)}


def execute(
    run: Run,
    classifier: Classifier,
    results_path: Path,
    runs_dir: Path,
    limit: int | None = None,
    data_dir: Path = data.DATA_DIR,
) -> dict | None:
    """Fit, predict, evaluate; returns the results row, or ``None`` if the table already has the run.

    ``limit`` scores the first ``limit`` test examples only, for a quick end-to-end check.
    """
    rid = run_id(run, classifier, limit, data_dir)
    if rid in completed(results_path):
        return None
    labels = data.labels(run.task, data_dir)
    train = data.subset(data.load(run.task, "train", data_dir), run.per_class, run.seed)
    test = data.load(run.task, run.split, data_dir)
    if limit:
        test = test.head(limit)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        (fit_s, train_n), train_peak_mb = in_fresh_process(
            _fit, classifier, run.task, train, labels, run.seed, out_dir
        )
        (probs, latency_ms, predict_s), peak_mb = in_fresh_process(
            _predict, classifier, out_dir, test["text"].tolist()
        )
        disk_mb = classifier.disk_mb(out_dir)

    probs = validate(np.asarray(probs, dtype=float), len(test), len(labels))
    run_dir = Path(runs_dir) / run.task / rid
    write_predictions(run_dir, test["id"], labels, probs, latency_ms)
    source = data.manifest(run.task, data_dir)["source"]
    row = {
        "run_id": rid,
        **{key: value for key, value in asdict(run).items() if key != "per_class"},
        "per_class": run.per_class or "all",
        "n_classes": len(labels),
        "chain": source.get("chain", ""),
        "data_version": data.data_version(run.task, data_dir),
        "train_n": train_n,
        "fit_s": round(fit_s, 2),
        "train_peak_mb": train_peak_mb,
        "predict_s": round(predict_s, 2),
        "peak_mb": peak_mb,
        "disk_mb": disk_mb,
        **evaluate(test["label"], labels, probs, latency_ms),
    }
    (run_dir / "run.json").write_text(
        json.dumps({**row, "classifier": classifier.identity()}, indent=2) + "\n"
    )
    append_row(row, results_path)
    return row


def _fit(classifier: Classifier, task: str, train: pd.DataFrame, labels: list[str], seed: int, out_dir: Path):
    start = time.perf_counter()
    train_n = classifier.fit(task, train, labels, seed, out_dir)
    return time.perf_counter() - start, train_n


def _predict(classifier: Classifier, out_dir: Path, texts: list[str]):
    """Batch size 1, one text at a time, as a service answering requests would."""
    start = time.perf_counter()
    predictor = classifier.load(out_dir)
    probs, latency_ms = [], []
    for i, text in enumerate(texts, 1):
        tick = time.perf_counter()
        probs.append(predictor.predict(text))
        latency_ms.append((time.perf_counter() - tick) * 1000)
        if i % 500 == 0:
            print(f"    {i}/{len(texts)}", flush=True)
    return np.array(probs), latency_ms, time.perf_counter() - start


def validate(probs: np.ndarray, n_texts: int, n_labels: int, tolerance: float = 1e-3) -> np.ndarray:
    """Reject anything that is not one probability distribution over the labels per text."""
    if probs.shape != (n_texts, n_labels):
        raise ValueError(f"expected probabilities of shape {(n_texts, n_labels)}, got {probs.shape}")
    if np.isnan(probs).any() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("probabilities must be in [0, 1]")
    if (np.abs(probs.sum(axis=1) - 1) > tolerance).any():
        raise ValueError("probabilities must sum to 1 for every text")
    return probs


def evaluate(gold_labels: pd.Series, labels: list[str], probs: np.ndarray, latency_ms: list[float]) -> dict:
    index = {label: i for i, label in enumerate(labels)}
    gold = gold_labels.map(index).to_numpy()
    pred = probs.argmax(axis=1)
    f1, _ = per_class_f1(gold, pred, len(labels))
    p50, p95 = np.percentile(latency_ms, [50, 95])
    return {
        "n": len(gold),
        **{key: round(float(value), 4) for key, value in scores(gold, pred, len(labels)).items()},
        "per_class_f1": json.dumps({label: round(float(v), 4) for label, v in zip(labels, f1, strict=True)}),
        "latency_p50_ms": round(float(p50), 3),
        "latency_p95_ms": round(float(p95), 3),
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write_predictions(
    run_dir: Path, ids: pd.Series, labels: list[str], probs: np.ndarray, latency_ms: list[float]
) -> None:
    """``predictions.csv``: id, predicted label, latency, then one ``p:<label>`` column per label."""
    run_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(probs, columns=[f"p:{label}" for label in labels])
    df.insert(0, "id", list(ids))
    df.insert(1, "predicted", [labels[i] for i in probs.argmax(axis=1)])
    df.insert(2, "latency_ms", latency_ms)
    df.to_csv(run_dir / "predictions.csv", index=False)


def append_row(row: dict, results_path: Path) -> None:
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    new = not results_path.exists()
    with results_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)
