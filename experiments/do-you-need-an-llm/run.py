"""Run every experiment of "Do You Need an LLM for That? Classifying Text Locally".

Five approaches on eleven tasks (VAST, DynaSent round 2, CLINC150 with out-of-scope, and eight CLINC150
slices of 10 to 100 intents), plus learning curves for the two trained approaches. Each run appends one row to
``results/results.csv``; runs already there are skipped, so the same command resumes after a stop.
Runs go cheapest first: TF-IDF, then the encoder, then the 4B models.

    uv run experiments/do-you-need-an-llm/run.py --dry-run          # list the runs
    uv run experiments/do-you-need-an-llm/run.py                    # run everything (about 15 hours)
    uv run experiments/do-you-need-an-llm/run.py --only tfidf ettin-150m --tasks vast
    uv run experiments/do-you-need-an-llm/run.py --limit 50         # quick end-to-end check
    touch experiments/do-you-need-an-llm/STOP                       # stop after the current run
"""

import argparse
import time
from pathlib import Path

from lc.data import CLASS_SLICES, CLINC150, slice_name
from lc.models import Classifier
from lc.models.decision import DecisionModel
from lc.models.encoder import FineTunedEncoder
from lc.models.prompted import PromptedLLM
from lc.models.tfidf import TfidfLogisticRegression
from lc.runs import Run, execute

HERE = Path(__file__).resolve().parent
PROMPTS = str(HERE / "prompts")
RESULTS_DIR = HERE / "results"
STOP_FILE = HERE / "STOP"

QWEN = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
APPROACHES: dict[str, Classifier] = {
    "tfidf": TfidfLogisticRegression(),
    "ettin-150m": FineTunedEncoder("jhu-clsp/ettin-encoder-150m", "45d08642849e5c5701b162671ac811b7654bfd9f"),
    "qwen3.5-4b-zero-shot": PromptedLLM(*QWEN, PROMPTS),
    "qwen3.5-4b-few-shot": PromptedLLM(*QWEN, PROMPTS, shots=(("clinc150", 1), ("vast", 5))),
    "onejev1.6-4b": DecisionModel("OmniJev/OneJev1.6-4B", "bb248003b6eb771a995cb3672ad47db090a9cd48"),
}
TRAINED = ["tfidf", "ettin-150m"]
# DynaSent was added after the other tasks had run. A prompted LLM's run id hashes every template of its
# prompt folder, so DynaSent's prompts live in a folder of their own, read by these classifiers only: the
# other tasks keep their run ids.
DYNASENT_PROMPTS = str(HERE / "prompts-dynasent")
FOR_TASK: dict[str, dict[str, Classifier]] = {
    "dynasent": {
        "qwen3.5-4b-zero-shot": PromptedLLM(*QWEN, DYNASENT_PROMPTS),
        "qwen3.5-4b-few-shot": PromptedLLM(*QWEN, DYNASENT_PROMPTS, shots=(("dynasent", 5),)),
    }
}

SLICES = [slice_name(n, chain) for n, chains in CLASS_SLICES.items() for chain in chains]
TASKS = [*SLICES, CLINC150, "vast", "dynasent"]
# Learning curves: training examples per class. On CLINC150 every intent has 100, so 100 is the full set.
CURVES = {
    CLINC150: [10, 25, 50],
    "vast": [10, 25, 50, 100, 333, 1000],
    "dynasent": [10, 25, 50, 100, 333, 1000],
}
# Small training sets vary most with the examples drawn: up to 100 per class, three draws.
CURVE_SEEDS = {size: (0, 1, 2) if size <= 100 else (0,) for sizes in CURVES.values() for size in sizes}


def plan(approaches: list[str], tasks: list[str]) -> list[Run]:
    runs = []
    for approach in approaches:
        runs += [Run(approach, task) for task in tasks]
        if approach in TRAINED:
            for task, sizes in CURVES.items():
                if task in tasks:
                    runs += [Run(approach, task, size, seed) for size in sizes for seed in CURVE_SEEDS[size]]
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", choices=list(APPROACHES), default=list(APPROACHES))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--dry-run", action="store_true", help="list the runs without running them")
    parser.add_argument("--limit", type=int, help="score the first N test examples only (separate results)")
    args = parser.parse_args()

    runs = plan(args.only, args.tasks)
    results = RESULTS_DIR / ("results.csv" if args.limit is None else f"check-{args.limit}.csv")
    runs_dir = RESULTS_DIR / ("runs" if args.limit is None else f"check-{args.limit}-runs")
    for i, run in enumerate(runs, 1):
        size = "all" if run.per_class is None else f"{run.per_class}/class"
        label = f"[{i}/{len(runs)}] {run.approach} {run.task} {size} seed {run.seed}"
        if args.dry_run:
            print(label)
            continue
        if STOP_FILE.exists():
            STOP_FILE.unlink()
            print(f"Stop file found: stopped before run {i} of {len(runs)}.")
            return
        print(f"{time.strftime('%H:%M:%S')} {label}", flush=True)
        classifier = FOR_TASK.get(run.task, {}).get(run.approach, APPROACHES[run.approach])
        row = execute(run, classifier, results, runs_dir, args.limit)
        if row is None:
            print("    already in the results, skipped", flush=True)
        else:
            print(
                f"    macro-F1 {row['macro_f1']:.4f}, p50 latency {row['latency_p50_ms']:.1f} ms", flush=True
            )
    print(f"{len(runs)} runs" if args.dry_run else f"Done. Results in {results}.")


if __name__ == "__main__":
    main()
