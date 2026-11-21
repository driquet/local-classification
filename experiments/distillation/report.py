"""The numbers of "No Labels? Borrow Them: Distilling a Classifier Locally", from ``results/``.

uv run experiments/distillation/report.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lc import data
from lc.metrics import macro_f1

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TASK = "arxiv"
LABELS = list(data.ARXIV_CATEGORIES)
# Teacher -> its approach name and the OpenRouter slug it is billed under (None: on the laptop).
TEACHERS = {
    "joey-1": ("joey-1:api", "baseten/joey-1"),
    "pplx-decider-flash-1.2": ("pplx-decider-flash-1.2:api", "perplexity/pplx-decider-flash-v1.2"),
    "onejev1.6-4b": ("onejev1.6-4b:bf16", None),
    "elephant-1": ("elephant-1:api", "baseten/elephant-1"),
    "mammoth-1": ("mammoth-1:api", "baseten/mammoth-1"),
    "pplx-decider-1.2": ("pplx-decider-1.2:api", "perplexity/pplx-decider-v1.2"),
}
BEST = "pplx-decider-1.2"
LABELED = 9000  # training pool and validation set
EXAMPLE = "Compact convolutional neural networks for AI-based drone detection system"


def read(name: str) -> pd.DataFrame:
    """A results table, without the runs of teachers the post dropped (Amber, OneJev1.6-9B)."""
    df = pd.read_csv(RESULTS / f"{name}.csv", dtype={"run_id": str})
    tasks = {TASK} | {f"{TASK}-silver-{teacher}" for teacher in TEACHERS}
    approaches = {approach for approach, _ in TEACHERS.values()} | {
        "tfidf",
        "student:gold",
        "onejev1.6-4b:q4_0:metal",
        *(f"student:{teacher}:{kind}" for teacher in TEACHERS for kind in ("soft", "hard")),
    }
    return df[df["task"].isin(tasks) & df["approach"].isin(approaches)]


def predictions(results: pd.DataFrame, approach: str, split: str = "test", seed: int = 0, prefix: str = ""):
    """A run's predictions, indexed by paper id, with the gold label alongside."""
    row = results[(results["approach"] == approach) & (results["split"] == split) & (results["seed"] == seed)]
    run_dir = RESULTS / f"{prefix}results-runs" / row["task"].iloc[-1] / row["run_id"].iloc[-1]
    df = pd.read_csv(run_dir / "predictions.csv", dtype={"id": str}).set_index("id")
    gold = data.load(TASK, split).set_index("id")["label"]
    return df.assign(gold=gold.loc[df.index])


def scores(results: pd.DataFrame) -> pd.DataFrame:
    """Test macro-F1 of every approach: mean, lowest and highest seed, number of seeds."""
    test = results[results["split"] == "test"]
    table = test.groupby(["approach", "task"])["macro_f1"].agg(["mean", "min", "max", "count"])
    return (table * [100, 100, 100, 1]).round(1)


def per_category(results: pd.DataFrame) -> pd.DataFrame:
    """F1 per category: the best teacher, its soft student and the gold encoder, means over seeds."""
    test = results[results["split"] == "test"]
    columns = {
        "teacher": TEACHERS[BEST][0],
        "student": f"student:{BEST}:soft",
        "gold": "student:gold",
    }
    table = {}
    for name, approach in columns.items():
        f1 = test.loc[test["approach"] == approach, "per_class_f1"].map(json.loads)
        table[name] = pd.DataFrame(f1.tolist()).mean() * 100
    return np.floor(pd.DataFrame(table) + 0.5)  # half up: 68.5 is 69


def mistakes(results: pd.DataFrame) -> None:
    """What students answer where their teacher is wrong, and how often each agrees with its teacher."""
    for teacher, (approach, _) in TEACHERS.items():
        student_approach = f"student:{teacher}:soft"
        if student_approach not in set(results["approach"]):
            continue
        t, s = predictions(results, approach), predictions(results, student_approach)
        wrong = t["predicted"] != t["gold"]
        same = (s["predicted"] == t["predicted"])[wrong].sum()
        fixed = (s["predicted"] == s["gold"])[wrong].sum()
        new = ((s["predicted"] != s["gold"]) & ~wrong).sum()
        n = wrong.sum()
        print(
            f"{teacher}: wrong on {n}; its student (seed 0): same mistake {same / n:.0%}, fixed {fixed} "
            f"({fixed / n:.0%}), another {(n - same - fixed) / n:.0%}; new mistakes {new}; "
            f"agreement {(s['predicted'] == t['predicted']).mean():.1%}"
        )


def misfiled(results: pd.DataFrame) -> None:
    """Where three teachers put the retrieval and security papers."""
    for teacher in ("onejev1.6-4b", "mammoth-1", BEST):
        t = predictions(results, TEACHERS[teacher][0])
        for category in ("cs.IR", "cs.CR"):
            counts = t.loc[t["gold"] == category, "predicted"].value_counts()
            print(f"{teacher}, gold {category}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))


def teachers(results: pd.DataFrame) -> pd.DataFrame:
    """Silver labels and costs: accuracy on the pool, mean top probability on the test set, labeling time
    (pool and validation), API cost and latency."""
    usage = pd.read_json(RESULTS / "api-usage.jsonl", lines=True).groupby("model")["cost"].mean()
    rows = {}
    for teacher, (approach, slug) in TEACHERS.items():
        runs = results[(results["approach"] == approach) & (results["task"] == TASK)].set_index("split")
        test = predictions(results, approach)
        labeling_s = runs.loc["train", "predict_s"] + runs.loc["validation", "predict_s"]
        cost = usage.get(slug, np.nan)
        rows[teacher] = {
            "pool accuracy": round(runs.loc["train", "accuracy"] * 100, 1),
            "mean top probability": round(test[[f"p:{label}" for label in LABELS]].max(axis=1).mean(), 2),
            "labeling min": round(labeling_s / 60),
            "labeling $": round(cost * LABELED, 3),
            "$ per million": round(cost * 1e6, 2),
            "latency ms": round(runs.loc["test", "latency_p50_ms"]),
        }
    return pd.DataFrame(rows).T


def pilot() -> pd.Series:
    """The teachers on the pilot's 300 papers (100 per split), against the gold labels."""
    results = read("check-100-results")
    out = {}
    for teacher, (approach, _) in TEACHERS.items():
        splits = ("validation", "test", "train")
        df = pd.concat(predictions(results, approach, split, prefix="check-100-") for split in splits)
        gold, pred = (df[c].map(LABELS.index).to_numpy() for c in ("gold", "predicted"))
        out[teacher] = round(macro_f1(gold, pred, len(LABELS)) * 100, 1)
    cost = pd.read_json(RESULTS / "check-100-api-usage.jsonl", lines=True)["cost"].sum()
    print(f"Pilot API cost: ${cost:.3f}")
    return pd.Series(out).sort_values()


def example(results: pd.DataFrame) -> None:
    """The paper of the post's first chart: OneJev's probabilities, and its soft student's answer."""
    test = data.load(TASK, "test").set_index("id")
    pid = test.index[test["text"].str.startswith(EXAMPLE)][0]
    teacher = predictions(results, TEACHERS["onejev1.6-4b"][0]).loc[pid]
    student = predictions(results, "student:onejev1.6-4b:soft").loc[pid]
    probs = teacher[[f"p:{label}" for label in LABELS]].astype(float).round(2).sort_values(ascending=False)
    print(f"{pid}, gold {teacher['gold']}: {probs.to_dict()}; soft student: {student['predicted']}")


def main() -> None:
    results = read("results")
    pd.set_option("display.width", 200)
    print("Test macro-F1\n", scores(results).to_string(), "\n", sep="")
    print("F1 per category\n", per_category(results).to_string(), "\n", sep="")
    mistakes(results)
    print()
    misfiled(results)
    print("\nTeachers\n", teachers(results).to_string(), "\n", sep="")
    served = pd.concat([results[results["split"] == "test"], read("latency")])
    serving = served.groupby("approach")[["fit_s", "latency_p50_ms", "peak_mb"]].mean().round(1)
    print("Training and serving (means over seeds)\n", serving.to_string(), "\n", sep="")
    print("Pilot macro-F1\n", pilot().to_string(), "\n", sep="")
    example(results)


if __name__ == "__main__":
    main()
