"""Datasets, frozen splits and training subsets.

Every task is prepared once and frozen in ``data/<task>/``: one JSONL file per split (``id``, ``text``,
``label``) and a ``manifest.json`` with the label list, the source and a SHA-256 checksum per split.
Every load verifies the checksums, so all approaches see exactly the same examples, and a change in an
upstream file fails loudly instead of silently changing the results.

Tasks:

- ``vast``: stance of a comment toward a topic (for, against, neutral), official splits.
- ``dynasent``: sentiment of a sentence (positive, negative, neutral), DynaSent round 2, official splits.
- ``clinc150``: 150 intents plus out-of-scope, official splits; the Decision Index protocol.
- ``clinc150-k<n>-c<chain>``: CLINC150 restricted to ``n`` in-scope intents. A chain is one seeded random
  order of the 150 intents, and a slice keeps the first ``n``, so within a chain every slice contains the
  smaller ones. Out-of-scope examples are dropped, so the class count is the only thing that changes.
"""

import functools
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SPLITS = ("train", "validation", "test")

VAST_COMMIT = "e7c4775182b184730f350995f8260579c9e066fe"
VAST_URL = (
    f"https://raw.githubusercontent.com/emilyallaway/zero-shot-stance/{VAST_COMMIT}/data/VAST/vast_{{}}.csv"
)
VAST_LABELS = {1: "for", 0: "against", 2: "neutral"}

DYNASENT_COMMIT = "a71b37a65108bcbd70a955ec786de1d636b3ad43"
DYNASENT_URL = f"https://github.com/cgpotts/dynasent/raw/{DYNASENT_COMMIT}/dynasent-v1.1.zip"
DYNASENT_LABELS = ["positive", "negative", "neutral"]

CLINC150_REVISION = "155b9c710419136e17307b80d0a13e68cd46b4ec"
CLINC150_URL = f"https://huggingface.co/datasets/clinc/clinc_oos/resolve/{CLINC150_REVISION}/{{}}"
CLINC150 = "clinc150"
OUT_OF_SCOPE = "oos"
# Class count -> chains. Small slices depend most on which intents are drawn, so they get three chains.
CLASS_SLICES = {10: (0, 1, 2), 25: (0, 1, 2), 50: (0,), 100: (0,)}

Splits = dict[str, pd.DataFrame]


def vast_text(topic: str, comment: str) -> str:
    """The single input every approach sees for VAST."""
    return f"Topic: {topic}\nComment: {comment}"


def build_vast() -> tuple[Splits, list[str], dict]:
    """Official train / dev (as validation) / test.

    The topic comes from ``topic_str``, the column the VAST authors' code reads. ``new_topic`` looks
    similar but, on the examples made neutral by pairing a comment with a topic from elsewhere, it still
    holds the comment's original topic: with it, every neutral example duplicates the input of a "for" or
    "against" one.
    """
    splits = {}
    for name, file in [("train", "train"), ("validation", "dev"), ("test", "test")]:
        df = _read_csv(VAST_URL.format(file))
        splits[name] = pd.DataFrame(
            {
                "id": [f"{name}-{i}" for i in df["new_id"]],
                "text": [vast_text(t, c) for t, c in zip(df["topic_str"], df["post"], strict=True)],
                "label": df["label"].map(VAST_LABELS),
            }
        )
    assert len(splits["test"]) == 3006, f"expected 3,006 VAST test examples, got {len(splits['test'])}"
    source = {"url": VAST_URL.format("{train,dev,test}"), "licence": "none stated in the source repository"}
    return splits, list(VAST_LABELS.values()), source


def build_dynasent() -> tuple[Splits, list[str], dict]:
    """Round 2, the adversarial one: crowdworkers wrote sentences that a strong sentiment model got wrong.

    Official train / dev (as validation) / test, keeping the sentences whose five validators reach a
    majority label (``gold_label``) among the three classes, as the authors' ternary version does: "mixed"
    and no-majority sentences are dropped from training; the test set has none.
    """
    with urllib.request.urlopen(DYNASENT_URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    splits = {}
    for name, file in [("train", "train"), ("validation", "dev"), ("test", "test")]:
        lines = archive.read(f"dynasent-v1.1/dynasent-v1.1-round02-dynabench-{file}.jsonl").splitlines()
        rows = [r for r in map(json.loads, lines) if r["gold_label"] in DYNASENT_LABELS]
        splits[name] = pd.DataFrame(
            {
                "id": [r["text_id"] for r in rows],
                "text": [r["sentence"] for r in rows],
                "label": [r["gold_label"] for r in rows],
            }
        )
    assert len(splits["test"]) == 720, f"expected 720 DynaSent test examples, got {len(splits['test'])}"
    return splits, DYNASENT_LABELS, {"url": DYNASENT_URL, "round": 2, "licence": "CC-BY-4.0"}


@functools.cache
def build_clinc150() -> tuple[Splits, list[str], dict]:
    """The ``plus`` configuration: 150 intents and out-of-scope, as the Decision Index scores it."""
    intents = _clinc150_intents()
    splits = {}
    for name in SPLITS:
        df = pd.read_parquet(CLINC150_URL.format(f"plus/{name}-00000-of-00001.parquet"))
        splits[name] = pd.DataFrame(
            {
                "id": [f"{name}-{i:05d}" for i in range(len(df))],
                "text": df["text"],
                "label": df["intent"].map(dict(enumerate(intents))),
            }
        )
    assert len(intents) == 151 and OUT_OF_SCOPE in intents, f"expected 151 labels, got {len(intents)}"
    assert len(splits["test"]) == 5500, f"expected 5,500 test examples, got {len(splits['test'])}"
    source = {"url": CLINC150_URL.format("plus/"), "licence": "CC-BY-3.0", "n_classes": 151}
    return splits, sorted(intents), source


def class_order(intents: list[str], chain: int) -> list[str]:
    """The chain's seeded random order of the intents (chain c uses seed c)."""
    return [intents[i] for i in np.random.default_rng(chain).permutation(len(intents))]


def class_slice(splits: Splits, intents: list[str], n_classes: int, chain: int) -> tuple[Splits, list[str]]:
    """Splits restricted to the first ``n_classes`` intents of the chain, and those intents in label order."""
    keep = set(class_order(intents, chain)[:n_classes])
    labels = [label for label in intents if label in keep]
    return {name: df[df["label"].isin(keep)] for name, df in splits.items()}, labels


def slice_name(n_classes: int, chain: int) -> str:
    return f"{CLINC150}-k{n_classes}-c{chain}"


def _slice_builder(n_classes: int, chain: int) -> Callable[[], tuple[Splits, list[str], dict]]:
    def build():
        splits, labels, source = build_clinc150()
        intents = [label for label in labels if label != OUT_OF_SCOPE]
        in_scope = {name: df[df["label"] != OUT_OF_SCOPE] for name, df in splits.items()}
        splits, labels = class_slice(in_scope, intents, n_classes, chain)
        return splits, labels, {**source, "n_classes": n_classes, "chain": chain}

    return build


BUILDERS: dict[str, Callable[[], tuple[Splits, list[str], dict]]] = {
    "vast": build_vast,
    "dynasent": build_dynasent,
    CLINC150: build_clinc150,
    **{slice_name(n, c): _slice_builder(n, c) for n, chains in CLASS_SLICES.items() for c in chains},
}


def prepare(task: str, data_dir: Path = DATA_DIR) -> None:
    """Download and freeze a task, unless it is already there. Rebuilding never changes frozen splits."""
    if (Path(data_dir) / task / "manifest.json").exists():
        return
    splits, labels, source = BUILDERS[task]()
    write_task(Path(data_dir) / task, splits, labels, source)


def write_task(task_dir: Path, splits: Splits, labels: list[str], source: dict) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"labels": list(labels), "source": source, "splits": {}}
    for name, df in splits.items():
        unknown = set(df["label"]) - set(labels)
        assert not unknown, f"{name}: labels missing from the label list: {sorted(unknown)}"
        path = task_dir / f"{name}.jsonl"
        rows = df.sort_values("id")[["id", "text", "label"]].to_dict("records")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        manifest["splits"][name] = {"n": len(rows), "sha256": _sha256(path)}
    (task_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def manifest(task: str, data_dir: Path = DATA_DIR) -> dict:
    prepare(task, data_dir)
    return json.loads((Path(data_dir) / task / "manifest.json").read_text())


def labels(task: str, data_dir: Path = DATA_DIR) -> list[str]:
    """The task's labels, in the fixed order used by every probability vector."""
    return manifest(task, data_dir)["labels"]


def data_version(task: str, data_dir: Path = DATA_DIR) -> str:
    """A short hash of the frozen splits: part of every run's identity."""
    splits = manifest(task, data_dir)["splits"]
    return hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()[:12]


def load(task: str, split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One split, after checking it against its frozen checksum."""
    expected = manifest(task, data_dir)["splits"][split]["sha256"]
    path = Path(data_dir) / task / f"{split}.jsonl"
    if _sha256(path) != expected:
        raise RuntimeError(f"{path}: checksum mismatch, the frozen split has changed")
    return pd.DataFrame([json.loads(line) for line in path.read_text().splitlines()])


def subset(train: pd.DataFrame, per_class: int | None, seed: int) -> pd.DataFrame:
    """A stratified subset with at most ``per_class`` examples per class (``None``: the whole set).

    Every example gets one seeded random rank, and each class keeps its lowest ranks. For a given seed,
    every subset is therefore contained in every larger one: a learning curve adds examples, it never
    swaps them.
    """
    if per_class is None:
        return train
    ranked = train.assign(_rank=np.random.default_rng(seed).permutation(len(train)))
    picked = ranked.sort_values("_rank").groupby("label").head(per_class)
    return picked.drop(columns="_rank").sort_values("id").reset_index(drop=True)


def few_shot_examples(task: str, per_class: int, seed: int, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Training examples for a few-shot prompt: ``per_class`` per label, in a seeded random order.

    CLINC150 slices draw from the full CLINC150 training set and keep their own intents, so an intent is
    illustrated by the same example in every slice. Otherwise the class-count curve would also be a curve
    of different draws. Examples are shuffled rather than grouped by class, which would favor the class
    shown last.
    """
    pool = CLINC150 if task.startswith(CLINC150) else task
    picked = subset(load(pool, "train", data_dir), per_class, seed)
    picked = picked[picked["label"].isin(labels(task, data_dir))]
    return picked.sample(frac=1, random_state=seed).reset_index(drop=True)


def _read_csv(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url) as response:
        return pd.read_csv(io.BytesIO(response.read()))


def _clinc150_intents() -> list[str]:
    """Intent names in class-id order, from the dataset card's ``plus`` configuration."""
    with urllib.request.urlopen(CLINC150_URL.format("README.md")) as response:
        card = response.read().decode()
    block = card.split("config_name: plus", 1)[1].split("config_name:", 1)[0]
    return [name.strip("'") for name in re.findall(r"^\s+'\d+': (.+)$", block, flags=re.M)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
