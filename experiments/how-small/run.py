"""Run every experiment of "How Small Can a Text Classifier Be?".

Each stage appends one row per run to a table in ``results/``. A run already in its table is skipped, so the
same command resumes after a stop. Stages, in the order they ran:

    sweep      learning rate of Ettin 17M, 32M and 68M, on validation splits; copy the winners into RECIPES
    tfidf      TF-IDF on the three tasks, to measure its memory in a process without torch
    full       Ettin 17M to 150M on the full training sets, seed 0, checkpoints kept
    curves     learning curves of Ettin 17M to 68M (150M's come from the first post)
    frozen     MiniLM and frozen Ettin-32M with a logistic regression, learning curves included
    serve      the checkpoints of "full" in fp16 and int8, and latency on the CPU and GPU
    seeds      seeds 1 and 2 of Ettin 17M to 150M on the full training sets
    llm        Qwen3.5-4B zero-shot and OneJev1.6 4B and 0.8B in 8 and 4 bits on the GPU (0.8B in bf16 too)
    sweep150   learning rate of Ettin-150M with the small models' 4 epochs, on validation splits
    recipe150  that Ettin-150M, on the full training sets and learning curves; it picks the sweep's winner
    embedders  GTE small and base, frozen, with a logistic regression, learning curves included
    seeds150   seeds 1 and 2 of that Ettin-150M, seed 1 kept and served in fp16 and int8 (one thread)
    gtecpu     latency and memory of the frozen GTE-base on one CPU thread
    gguf       the 4B models on llama.cpp: latency and memory, Q8_0 and Q4_0 on the GPU, Q4_0 on 4 CPU threads
    gguffull   llama.cpp's Q4_0 on the GPU, on the full test sets, for macro-F1 next to MLX's 4 bits
    gguf8cpu   llama.cpp's Q8_0 on 4 CPU threads: latency and memory
    llmcpu     the LLMs on the CPU, DynaSent only: llama.cpp in bf16 (4 threads), MLX in bf16, 8 and 4 bits
    ettin400   Ettin-400M (gradient checkpointing, 3e-5, unswept): full training sets, serving variants and
               DynaSent's learning curve
    amber      Amber decision encoders 32M to 400M: bf16 and fp32 on the GPU and int8 on the CPU on the full
               test sets, latency of fp32 and int8 on one CPU thread

    uv run experiments/how-small/run.py --dry-run                          # list the runs
    caffeinate -i -s uv run experiments/how-small/run.py --stages sweep    # then fill RECIPES
    caffeinate -i -s uv run experiments/how-small/run.py --stages tfidf full curves
    uv run experiments/how-small/run.py --stages full serve --only ettin-17m --tasks dynasent --limit 50
    touch experiments/how-small/STOP                                       # stop after the current run

The README lists the commands that produced the published results. An approach reads ``model:variant``, for
instance ``ettin-17m:int8:cpu4``. Importing this file loads no machine learning library: each process spawned
for a run imports only what its approach needs, so the memory of a CPU predictor does not include torch and
MLX.
"""

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from lc.measure import in_fresh_process
from lc.runs import Run, execute

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
# MLX quantizations, llama.cpp weights (README, "llama.cpp") and Amber's int8 export.
MODELS_DIR = RESULTS_DIR / "models"
STOP_FILE = HERE / "STOP"
FIRST_POST = HERE.parent / "do-you-need-an-llm"

ENCODERS = {
    "ettin-17m": ("jhu-clsp/ettin-encoder-17m", "987607455c61e7a5bbc85f7758e0512ea6d0ae4c"),
    "ettin-32m": ("jhu-clsp/ettin-encoder-32m", "1b8ba06455dd44f80fc9c1ca9e22806157a57379"),
    "ettin-68m": ("jhu-clsp/ettin-encoder-68m", "ac19ae4bc51093b31c475665ac872a936d056cc2"),
    "ettin-150m": ("jhu-clsp/ettin-encoder-150m", "45d08642849e5c5701b162671ac811b7654bfd9f"),
}
# Too big for the size sweep's budget: trained with gradient checkpointing, without which VAST needs about
# 47 GB, and a learning rate picked, not swept (3e-5, below 150M's 5e-5, as larger encoders usually need).
LARGE = {"ettin-400m": ("jhu-clsp/ettin-encoder-400m", "7662476d60abb071a5bd319c9f3074f3072c062d")}
# Ettin-150M again, with the small models' recipe: 4 epochs and its own swept learning rate.
TUNED = {"ettin-150m-4ep": "ettin-150m"}
EMBEDDERS = {
    "minilm-l6": ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    "gte-small": ("thenlper/gte-small", "17e1f347d17fe144873b1201da91788898c639cd"),
    "gte-base": ("thenlper/gte-base", "c078288308d8dee004ab72c6191778064285ec0c"),
}
LLMS = {
    "qwen3.5-4b-zero-shot": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "onejev1.6-4b": ("OmniJev/OneJev1.6-4B", "bb248003b6eb771a995cb3672ad47db090a9cd48"),
    "onejev1.6-0.8b": ("OmniJev/OneJev1.6-0.8B", "bcd88a382f314b26cb7b1b7f369c1ed3502ee7a0"),
}
# Decision models built on Ettin rerankers, from the Decision Index (CC BY-NC 4.0). Trained on CLINC150's
# training split, among others: on CLINC150 they are not zero-shot.
AMBER = {
    "amber-32m": ("MLukauskas/amber-32m", "15ba6d8873e01773d1d9ec1b976dc21d17779002"),
    "amber-68m": ("MLukauskas/amber-68m", "40cd609452ca2642a66a19d6476f6919bad318f0"),
    "amber-150m": ("MLukauskas/amber-150m", "51ee1bba1f9e71aae4bf65bde1d43216260b6aab"),
    "amber-400m": ("MLukauskas/amber-0.4b-v2", "38d82ac41c0391a767b10c604f0d6e9b6773e910"),
}
# Their bf16 runs on the GPU come from the first post; the others' come from the llm stage.
IN_FIRST_POST = {"qwen3.5-4b-zero-shot", "onejev1.6-4b"}

# Ettin-150M keeps the recipe of the first post (3 epochs, 5e-5), so that post's learning curve still
# applies. The others get the sweep's winner.
SMALL = ["ettin-17m", "ettin-32m", "ettin-68m"]
RECIPES: dict[str, dict] = {
    "ettin-17m": {"learning_rate": 1e-4, "epochs": 4},
    "ettin-32m": {"learning_rate": 5e-5, "epochs": 4},
    "ettin-68m": {"learning_rate": 5e-5, "epochs": 4},
    "ettin-150m": {},
    "ettin-400m": {"learning_rate": 3e-5, "epochs": 4, "gradient_checkpointing": True},
}
SWEEP_EPOCHS = 4
SWEEP_RATES = ["5e-5", "1e-4", "3e-4"]
SWEEP_RATES_150M = ["3e-5", "5e-5", "1e-4"]  # 68M already preferred the lowest rate on VAST

TASKS = ["clinc150", "vast", "dynasent"]
CURVES = {
    "clinc150": [10, 25, 50],
    "vast": [10, 25, 50, 100, 333, 1000],
    "dynasent": [10, 25, 50, 100, 333, 1000],
}
FROZEN = ["minilm-l6:frozen", "ettin-32m:frozen"]
SERVED = ["fp16:mps", "int8:cpu4"]  # scored on the full test sets
PROBED = ["fp32:cpu1", "fp32:cpu4", "int8:cpu1", "fp32:mps"]  # latency and memory only
AMBER_SERVED = ["bf16:mps", "fp32:mps", "int8:cpu4"]
AMBER_PROBED = ["fp32:cpu1", "int8:cpu1"]
PROBE_TEXTS = 1000  # enough for stable latency percentiles; predictions do not change with the device
LLM_PROBE_TEXTS = 200  # a 4B model on the CPU takes up to 16 s per CLINC150 text
MLX_QUANTIZED_CPU_TEXTS = 10  # MLX's CPU code for quantized weights took about 7 min per DynaSent text


@dataclass(frozen=True)
class Job:
    run: Run
    table: str = "results"
    limit: int | None = None
    keep: bool = False  # keep the trained weights, for the serving variants


def probe(approach: str, task: str, texts: int = PROBE_TEXTS) -> Job:
    return Job(Run(approach, task), "latency", texts)


def curve(approach: str, task: str, extra: tuple[int, ...] = ()) -> list[Job]:
    """A learning curve: three seeds up to 100 examples per class, one above."""
    sizes = [*extra, *CURVES[task]]
    return [Job(Run(approach, task, n, s)) for n in sizes for s in ((0, 1, 2) if n <= 100 else (0,))]


def full_and_curve(approach: str, task: str, extra: tuple[int, ...] = ()) -> list[Job]:
    return [Job(Run(approach, task)), *curve(approach, task, extra)]


def sweep(models: list[str], rates: list[str], task: str) -> list[Job]:
    """Validation runs, on CLINC150 and VAST (1,000 examples per class) only."""
    if task == "dynasent":
        return []
    per_class = None if task == "clinc150" else 1000
    return [
        Job(Run(f"{model}:lr{rate}", task, per_class, split="validation"), "sweep")
        for model in models
        for rate in rates
    ]


def serve(models: list[str], task: str) -> list[Job]:
    scored = [Job(Run(f"{model}:{variant}", task)) for model in models for variant in SERVED]
    return scored + [probe(f"{model}:{variant}", task) for model in models for variant in PROBED]


def serve_stage(task: str) -> list[Job]:
    return serve(list(ENCODERS), task) + [probe(f"minilm-l6:frozen:{cpu}", task) for cpu in ("cpu1", "cpu4")]


def llm_stage(task: str) -> list[Job]:
    """On the GPU. The CPU runs are the llmcpu stage."""
    return [
        Job(Run(f"{model}:{variant}", task))
        for model in LLMS
        for variant in ([] if model in IN_FIRST_POST else ["bf16"]) + ["8bit", "4bit"]
    ]


def seeds150_stage(task: str) -> list[Job]:
    """Seed 0 kept no checkpoint, so seed 1's is the one served."""
    return [
        job
        for model in TUNED
        for job in [
            Job(Run(model, task, seed=1), keep=True),
            Job(Run(model, task, seed=2)),
            *(Job(Run(f"{model}:{variant}", task, seed=1)) for variant in ("fp16:mps", "int8:cpu1")),
        ]
    ]


def gguf_stage(task: str, variants: list[str]) -> list[Job]:
    return [probe(f"{model}:{variant}", task, LLM_PROBE_TEXTS) for variant in variants for model in LLMS]


def llmcpu_stage(task: str) -> list[Job]:
    """DynaSent only: its short texts and 3 labels make the cheapest task. The quantized MLX runs, the
    slowest by far, come last and score a few texts."""
    if task != "dynasent":
        return []
    runs = [("bf16:cpu4", LLM_PROBE_TEXTS), ("bf16:cpu", LLM_PROBE_TEXTS)]
    runs += [(f"{bits}bit:cpu", MLX_QUANTIZED_CPU_TEXTS) for bits in (8, 4)]
    return [probe(f"{model}:{variant}", task, n) for variant, n in runs for model in LLMS]


def ettin400_stage(task: str) -> list[Job]:
    model = "ettin-400m"
    curves = curve(model, task) if task == "dynasent" else []
    return [Job(Run(model, task), keep=True), *serve([model], task), *curves]


def amber_stage(task: str) -> list[Job]:
    scored = [Job(Run(f"{model}:{variant}", task)) for variant in AMBER_SERVED for model in AMBER]
    return scored + [probe(f"{model}:{variant}", task) for variant in AMBER_PROBED for model in AMBER]


STAGES: dict[str, Callable[[str], list[Job]]] = {
    "sweep": lambda task: sweep(SMALL, SWEEP_RATES, task),
    "tfidf": lambda task: [Job(Run("tfidf", task))],
    "full": lambda task: [Job(Run(model, task), keep=True) for model in ENCODERS],
    "curves": lambda task: [job for model in SMALL for job in curve(model, task)],
    "frozen": lambda task: [job for approach in FROZEN for job in full_and_curve(approach, task, (5,))],
    "serve": serve_stage,
    "seeds": lambda task: [Job(Run(model, task, seed=seed)) for model in ENCODERS for seed in (1, 2)],
    "llm": llm_stage,
    "sweep150": lambda task: sweep(list(TUNED), SWEEP_RATES_150M, task),
    "recipe150": lambda task: [
        job for model in TUNED for job in [*curve(model, task), Job(Run(model, task))]
    ],
    "embedders": lambda task: [
        job for model in ("gte-small", "gte-base") for job in full_and_curve(f"{model}:frozen", task, (5,))
    ],
    "seeds150": seeds150_stage,
    "gtecpu": lambda task: [probe("gte-base:frozen:cpu1", task)],
    "gguf": lambda task: gguf_stage(task, ["q8_0:metal", "q4_0:metal", "q4_0:cpu4"]),
    "gguffull": lambda task: [Job(Run(f"{model}:q4_0:metal", task)) for model in LLMS],
    # Q8_0 on the CPU is its own stage: no faster than Q4_0 there in the smoke runs, and more memory.
    "gguf8cpu": lambda task: gguf_stage(task, ["q8_0:cpu4"]),
    "llmcpu": llmcpu_stage,
    "ettin400": ettin400_stage,
    "amber": amber_stage,
}
DEFAULT_STAGES = ["tfidf", "full", "curves", "frozen", "serve", "seeds"]


def plan(stages: list[str], tasks: list[str], only: list[str] | None = None) -> list[Job]:
    jobs = [job for name in stages for task in tasks for job in STAGES[name](task)]
    return [job for job in jobs if not only or job.run.approach.startswith(tuple(only))]


def swept(model: str) -> dict:
    """The learning rate with the best mean macro-F1 over the sweep's two tasks, once both are done."""
    import pandas as pd

    rows = pd.read_csv(RESULTS_DIR / "sweep.csv")
    scores = rows[rows["approach"].str.startswith(f"{model}:lr")].groupby("approach")["macro_f1"]
    if scores.ngroups < len(SWEEP_RATES_150M) or (scores.count() < 2).any():
        raise SystemExit(f"The sweep of {model} is not done: run the sweep150 stage first.")
    return {"learning_rate": float(scores.mean().idxmax().split(":lr")[1]), "epochs": SWEEP_EPOCHS}


def prompts(task: str) -> Path:
    """The first post's zero-shot prompts; DynaSent has its own."""
    return FIRST_POST / ("prompts-dynasent" if task == "dynasent" else "prompts")


def threads(device: str) -> int:
    """``cpu4`` -> 4."""
    return int(device.removeprefix("cpu"))


def fine_tuned(model: str, **recipe):
    from lc.models.encoder import FineTunedEncoder

    if not recipe and model not in RECIPES:
        if model not in TUNED:
            raise SystemExit(f"No recipe for {model}: run the sweep stage and fill RECIPES in {__file__}.")
        RECIPES[model] = swept(model)
    return FineTunedEncoder(*{**ENCODERS, **LARGE}[TUNED.get(model, model)], **(recipe or RECIPES[model]))


def served(model: str, precision: str, device: str, checkpoints: Path):
    """A fine-tuned checkpoint served in ``precision`` (fp32, fp16, int8) on ``mps`` or ``cpu<threads>``."""
    from lc.models.encoder import ServedEncoder

    base, saved = fine_tuned(model), str(checkpoints / model)
    if precision == "int8":
        from lc.models.onnx import OnnxEncoder

        return OnnxEncoder(json.dumps(base.identity()), saved, base.max_length, threads(device))
    if device == "mps":
        return ServedEncoder(base, saved, half=precision == "fp16")
    return ServedEncoder(base, saved, "cpu", threads(device))


def frozen(model: str, device: str | None):
    from lc.models.frozen import FrozenEncoder

    repo = EMBEDDERS.get(model) or ENCODERS[model]
    # Regularization picked on validation splits for MiniLM and Ettin; GTE, also a sentence encoder, gets
    # MiniLM's. MiniLM truncates at its own limit of 256 tokens.
    options = {"c": 100.0} if model in ENCODERS else {"c": 10.0}
    if model == "minilm-l6":
        options["max_length"] = 256
    if device:
        options |= {"device": "cpu", "threads": threads(device)}
    return FrozenEncoder(*repo, **options)


def quantized(model: str, bits: int) -> Path:
    """MLX weights in ``bits`` bits, quantized from the pinned revision on first use."""
    from lc.models.decision import quantize

    path = MODELS_DIR / f"{model}-{bits}bit"
    if not (path / "config.json").exists():
        in_fresh_process(quantize, *LLMS[model], bits, path)
    return path


def gguf(model: str, task: str, quant: str, device: str):
    """A 4B model on llama.cpp: ``quant`` names the weights file, ``device`` is ``metal`` or ``cpu<n>``."""
    from lc.models.gguf import GgufDecisionModel, GgufPromptedLLM

    repo, revision = LLMS[model]
    path = MODELS_DIR / f"{repo.split('/')[1].lower()}-{quant.upper()}.gguf"
    if not path.exists():
        raise SystemExit(f"{path} is missing: convert and quantize it as the README says.")
    if model.startswith("onejev"):
        return GgufDecisionModel(repo, revision, gguf=str(path), device=device)
    return GgufPromptedLLM(repo, revision, str(prompts(task)), gguf=str(path), device=device)


def llm(model: str, task: str, precision: str, cpu: bool = False):
    """A 4B model on MLX: ``precision`` is ``bf16`` (the pinned weights), ``8bit`` or ``4bit``."""
    from lc.models.decision import DecisionModel
    from lc.models.prompted import OnCpu, PromptedLLM

    weights = LLMS[model] if precision == "bf16" else (str(quantized(model, int(precision[0]))), None)
    if model.startswith("onejev"):
        classifier = DecisionModel(*weights)
    else:
        classifier = PromptedLLM(*weights, str(prompts(task)))
    return OnCpu(classifier) if cpu else classifier


def amber(model: str, precision: str, device: str):
    from lc.models.amber import AmberDecisionModel, AmberInt8

    n = threads(device) if device.startswith("cpu") else None
    if precision == "int8":
        return AmberInt8(*AMBER[model], str(MODELS_DIR / "amber-int8"), n)
    return AmberDecisionModel(*AMBER[model], "cpu" if n else device, n, precision)


def classifier(approach: str, task: str, checkpoints: Path):
    """The classifier an approach name stands for. Models load later, in the run's own processes."""
    model, *variant = approach.split(":")
    if model == "tfidf":
        from lc.models.tfidf import TfidfLogisticRegression

        return TfidfLogisticRegression()
    if model in AMBER:
        return amber(model, *variant)
    if model in LLMS:
        # llama.cpp names its device (metal, cpu<threads>); MLX runs on the GPU, or on the CPU as "cpu".
        precision, *device = variant
        if device and device[0] != "cpu":
            return gguf(model, task, precision, device[0])
        return llm(model, task, precision, cpu=bool(device))
    if variant[:1] == ["frozen"]:
        return frozen(model, variant[1] if variant[1:] else None)
    if not variant:
        return fine_tuned(model)
    if variant[0].startswith("lr"):
        return fine_tuned(model, learning_rate=float(variant[0][2:]), epochs=SWEEP_EPOCHS)
    return served(model, *variant, checkpoints)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stages", nargs="+", choices=list(STAGES), default=DEFAULT_STAGES)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--only", nargs="+", help="keep approaches starting with one of these")
    parser.add_argument("--dry-run", action="store_true", help="list the runs without running them")
    parser.add_argument("--limit", type=int, help="score the first N test examples only (separate results)")
    args = parser.parse_args()

    prefix = "" if args.limit is None else f"check-{args.limit}-"
    checkpoints = RESULTS_DIR / f"{prefix}checkpoints"
    jobs = plan(args.stages, args.tasks, args.only)
    for i, job in enumerate(jobs, 1):
        run = job.run
        size = "all" if run.per_class is None else f"{run.per_class}/class"
        label = f"[{i}/{len(jobs)}] {run.approach} {run.task} {size} seed {run.seed} ({job.table})"
        if args.dry_run:
            print(label)
            continue
        if STOP_FILE.exists():
            STOP_FILE.unlink()
            print(f"Stop file found: stopped before run {i} of {len(jobs)}.")
            return
        print(f"{time.strftime('%H:%M:%S')} {label}", flush=True)
        limit = min(filter(None, (job.limit, args.limit)), default=None)
        kept = checkpoints / run.approach / run.task / f"seed-{run.seed}" if job.keep else None
        row = execute(
            run,
            classifier(run.approach, run.task, checkpoints),
            RESULTS_DIR / f"{prefix}{job.table}.csv",
            RESULTS_DIR / f"{prefix}{job.table}-runs",
            limit,
            checkpoint=kept,
        )
        if row is None:
            print("    already in the results, skipped", flush=True)
        else:
            f1, p50 = row["macro_f1"], row["latency_p50_ms"]
            print(f"    macro-F1 {f1:.4f}, p50 latency {p50:.1f} ms", flush=True)
    print(f"{len(jobs)} runs" if args.dry_run else "Done.")


if __name__ == "__main__":
    main()
