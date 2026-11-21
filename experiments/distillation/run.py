"""Run every experiment of "No Labels? Borrow Them: Distilling a Classifier Locally".

A teacher (a decision model, no labels) labels the arXiv training pool and validation set once. A student
(Ettin-150M, part 2's recipe) is fine-tuned on its answers: hard (cross-entropy on its top answer) or soft
(KL divergence to its probabilities at T = 2). Everyone is graded on the gold test set.

    label     each teacher predicts the validation set, the test set and the training pool
    serve     latency and memory of OneJev1.6-4B as served: Q4_0 on the GPU (llama.cpp)
    tfidf     TF-IDF on the gold labels and on each teacher's
    students  the gold ceiling and each teacher's students, seeds as in the post

    uv run experiments/distillation/run.py --dry-run
    uv run experiments/distillation/run.py --pilot            # the teachers on 100 texts per split
    caffeinate -i -s uv run experiments/distillation/run.py
    touch experiments/distillation/STOP                       # stop after the current run

A teacher's run on the training split is its labeling: ``predict_s`` is the labeling time, and its accuracy
against the pool's gold labels is the quality of its silver labels. Its students train on the task
``arxiv-silver-<teacher>``, written from those predictions: the training and validation splits carry the
teacher's top answer as ``label`` and its probabilities as ``probs``; the test split keeps the gold labels.
"""

import argparse
import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lc import data
from lc.runs import Run, execute

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
STOP_FILE = HERE / "STOP"
TASK = "arxiv"
PILOT_TEXTS = 100
PROBE_TEXTS = 200

# Part 2's pinned models and loaders: OneJev1.6-4B (MLX and llama.cpp), Ettin-150M, TF-IDF.
_spec = importlib.util.spec_from_file_location("how_small", HERE.parent / "how-small" / "run.py")
how_small = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(how_small)

# Decision models on OpenRouter's Decisions API, two families by size, and OneJev1.6-4B on the laptop.
API_TEACHERS = {
    "joey-1": "baseten/joey-1",
    "elephant-1": "baseten/elephant-1",
    "mammoth-1": "baseten/mammoth-1",
    "pplx-decider-flash-1.2": "perplexity/pplx-decider-flash-v1.2",
    "pplx-decider-1.2": "perplexity/pplx-decider-v1.2",
}
TEACHERS = {"onejev1.6-4b": "onejev1.6-4b:bf16"} | {name: f"{name}:api" for name in API_TEACHERS}

# Part 2's Ettin-150M recipe. Batches of 16 abstracts at 512 tokens exceed the 15 GB MPS cap: each batch runs
# in 4 parts of 4, for the same update.
STUDENT = {"learning_rate": 5e-5, "epochs": 4, "accumulation": 4}
TEMPERATURE = 2.0
# Seeds per student: three where the post compares close numbers, fewer elsewhere. Joey 1 gets none.
SEEDS = {
    ("onejev1.6-4b", "soft"): 2,
    ("onejev1.6-4b", "hard"): 2,
    ("pplx-decider-1.2", "soft"): 3,
    ("pplx-decider-1.2", "hard"): 3,
    ("mammoth-1", "soft"): 3,
    ("mammoth-1", "hard"): 3,
    ("elephant-1", "soft"): 1,
    ("pplx-decider-flash-1.2", "soft"): 1,
}
GOLD_SEEDS = 3
DISTILLED = list(dict.fromkeys(teacher for teacher, _ in SEEDS))


def silver(teacher: str) -> str:
    return f"{TASK}-silver-{teacher}"


@dataclass(frozen=True)
class Job:
    run: Run
    table: str = "results"
    limit: int | None = None


STAGES = {
    "label": lambda: [
        Job(Run(approach, TASK, split=split))
        for approach in TEACHERS.values()
        for split in ("validation", "test", "train")
    ],
    "serve": lambda: [Job(Run("onejev1.6-4b:q4_0:metal", TASK), "latency", PROBE_TEXTS)],
    "tfidf": lambda: [Job(Run("tfidf", task)) for task in [TASK, *map(silver, DISTILLED)]],
    "students": lambda: (
        [Job(Run("student:gold", TASK, seed=s)) for s in range(GOLD_SEEDS)]
        + [
            Job(Run(f"student:{teacher}:{kind}", silver(teacher), seed=s))
            for (teacher, kind), n in SEEDS.items()
            for s in range(n)
        ]
    ),
}


def write_silver(teacher: str) -> None:
    """``arxiv-silver-<teacher>``: the teacher's answers on training and validation, gold on test."""
    task_dir = data.DATA_DIR / silver(teacher)
    if (task_dir / "manifest.json").exists():
        return
    results = pd.read_csv(RESULTS_DIR / "results.csv", dtype={"run_id": str})
    labels = data.labels(TASK)
    splits = {"test": data.load(TASK, "test")}
    for split in ("train", "validation"):
        row = results[(results["approach"] == TEACHERS[teacher]) & (results["split"] == split)]
        if row.empty:
            raise SystemExit(f"{teacher} has not labeled the {split} split: run the label stage first.")
        # arXiv ids look like numbers: read as floats, 2603.01230 would lose its last digit.
        path = RESULTS_DIR / "results-runs" / TASK / row["run_id"].iloc[-1] / "predictions.csv"
        predicted = pd.read_csv(path, dtype={"id": str}).set_index("id")
        gold = data.load(TASK, split)
        probs = predicted.loc[gold["id"], [f"p:{label}" for label in labels]].to_numpy()
        splits[split] = gold.assign(
            label=[labels[i] for i in probs.argmax(axis=1)],
            probs=[[round(float(p), 6) for p in row] for row in probs],
        )
    source = {"teacher": TEACHERS[teacher], "gold": data.manifest(TASK)["source"]}
    data.write_task(task_dir, splits, labels, source)


def classifier(approach: str, task: str, prefix: str):
    """The classifier an approach name stands for. Models load later, in the run's own processes."""
    if approach.endswith(":api"):
        from lc.models.decision_api import DecisionAPI

        usage_log = RESULTS_DIR / f"{prefix}api-usage.jsonl"
        return DecisionAPI(API_TEACHERS[approach.removesuffix(":api")], str(usage_log))
    if approach.startswith("student:"):
        from lc.models.encoder import FineTunedEncoder

        soft = {"soft": True, "temperature": TEMPERATURE} if approach.endswith(":soft") else {}
        return FineTunedEncoder(*how_small.ENCODERS["ettin-150m"], **STUDENT, **soft)
    return how_small.classifier(approach, task, RESULTS_DIR / "checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(STAGES))
    parser.add_argument("--only", nargs="+", help="keep approaches starting with one of these")
    parser.add_argument("--dry-run", action="store_true", help="list the runs without running them")
    parser.add_argument(
        "--pilot", action="store_true", help=f"label stage on {PILOT_TEXTS} texts per split, separate results"
    )
    args = parser.parse_args()

    stages = ["label"] if args.pilot else args.stages
    limit = PILOT_TEXTS if args.pilot else None
    prefix = f"check-{PILOT_TEXTS}-" if args.pilot else ""
    jobs = [job for name in stages for job in STAGES[name]()]
    jobs = [job for job in jobs if not args.only or job.run.approach.startswith(tuple(args.only))]
    failed = []
    for i, job in enumerate(jobs, 1):
        run = job.run
        label = f"[{i}/{len(jobs)}] {run.approach} {run.task} {run.split} seed {run.seed} ({job.table})"
        if args.dry_run:
            print(label)
            continue
        if STOP_FILE.exists():
            STOP_FILE.unlink()
            print(f"Stop file found: stopped before run {i} of {len(jobs)}.")
            return
        print(f"{time.strftime('%H:%M:%S')} {label}", flush=True)
        if run.task != TASK:
            write_silver(run.task.removeprefix(f"{TASK}-silver-"))
        try:
            row = execute(
                run,
                classifier(run.approach, run.task, prefix),
                RESULTS_DIR / f"{prefix}{job.table}.csv",
                RESULTS_DIR / f"{prefix}{job.table}-runs",
                job.limit or limit,
            )
        except RuntimeError as error:  # out of memory or an API failure: one run lost, not the night
            failed.append(label)
            print(f"    FAILED: {str(error).splitlines()[0]}", flush=True)
            continue
        if row is None:
            print("    already in the results, skipped", flush=True)
        else:
            print(f"    macro-F1 {row['macro_f1']:.4f}, p50 latency {row['latency_p50_ms']:.1f} ms, "
                  f"fit {row['fit_s']:.0f} s, predict {row['predict_s']:.0f} s", flush=True)  # fmt: skip
    if failed:
        print("Failed runs, rerun them with the same command:\n" + "\n".join(failed))
    print(f"{len(jobs)} runs" if args.dry_run else "Done.")


if __name__ == "__main__":
    main()
