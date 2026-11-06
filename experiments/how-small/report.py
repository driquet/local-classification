"""The tables of "How Small Can a Text Classifier Be?", from ``results/`` and the first post's results.

uv run experiments/how-small/report.py          # also prints the sweep's winners, for RECIPES in run.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIRST_POST = HERE.parent / "do-you-need-an-llm" / "results" / "results.csv"
TASKS = ["clinc150", "vast", "dynasent"]
BASELINES = ["ettin-150m", "qwen3.5-4b-zero-shot", "onejev1.6-4b", "tfidf"]
COST = [
    "macro_f1",
    "train_n",
    "fit_s",
    "disk_mb",
    "peak_mb",
    "gpu_used_mb",
    "latency_p50_ms",
    "latency_p95_ms",
]


def read(name: str) -> pd.DataFrame:
    path = RESULTS / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def sweep() -> None:
    table = read("sweep")
    if table.empty:
        return
    table[["model", "rate"]] = table["approach"].str.split(":", expand=True)
    scores = table.pivot_table(index=["model", "rate"], columns="task", values="macro_f1")
    scores["mean"] = scores.mean(axis=1)
    print("Sweep (validation macro-F1)\n", scores.round(4), sep="")
    print("\nWinners:\n", scores["mean"].groupby("model").idxmax().map(lambda key: key[1]), "\n", sep="")


def tables() -> None:
    results = read("results")
    first = pd.read_csv(FIRST_POST)
    first = first[first["task"].isin(TASKS) & first["approach"].isin(BASELINES)].assign(post="first")
    # A fresh run replaces nothing: Ettin-150M's two seeds 0 show how far one training can move.
    merged = pd.concat([results.assign(post="this"), first], ignore_index=True)

    for task in TASKS:
        rows = merged[(merged["task"] == task) & (merged["per_class"] == "all")]
        cost = rows.groupby(["approach", "post"])[COST].mean()
        cost.insert(1, "sd", rows.groupby(["approach", "post"])["macro_f1"].std())
        cost.insert(2, "seeds", rows.groupby(["approach", "post"])["seed"].nunique())
        print(f"{task}: full training set\n", cost.sort_values("disk_mb").round(4).to_string(), "\n", sep="")

        curves = merged[(merged["task"] == task) & (merged["per_class"] != "all")].copy()
        curves["per_class"] = curves["per_class"].astype(int)
        print(
            f"{task}: macro-F1 by examples per class\n",
            curves.pivot_table(index="approach", columns="per_class", values="macro_f1").round(4).to_string(),
            "\n",
            sep="",
        )

    latency = read("latency")
    if not latency.empty:
        print(
            "Latency probes (p50 ms)\n",
            latency.pivot_table(index="approach", columns="task", values="latency_p50_ms")
            .round(2)
            .to_string(),
            "\n\nServing memory, physical footprint (MB)\n",
            latency.pivot_table(index="approach", columns="task", values="peak_mb").round(0).to_string(),
            "\n\nGPU memory used by tensors (MB)\n",
            latency.pivot_table(index="approach", columns="task", values="gpu_used_mb").round(0).to_string(),
            sep="",
        )


def runtimes(texts: int = 200) -> None:
    """The 4B models on MLX and llama.cpp, latency over the same first ``texts`` test texts.

    MLX ran the full test sets one text at a time, so its first texts are read from its predictions. A
    llama.cpp run on the full test set also gives its agreement with MLX at the same number of bits.
    """
    results, latency = read("results"), read("latency")
    if latency.empty or not latency["approach"].str.contains(":q[48]_0:").any():
        return
    llms = ("qwen3.5-4b-zero-shot:", "onejev1.6-4b:", "onejev1.6-0.8b:")
    runs = pd.concat([results.assign(table="results"), latency.assign(table="latency")], ignore_index=True)
    runs = runs[runs["approach"].str.startswith(llms) & (runs["per_class"] == "all")]
    rows = []
    for run in runs.itertuples():
        predictions = pd.read_csv(RESULTS / f"{run.table}-runs" / run.task / run.run_id / "predictions.csv")
        first = predictions["latency_ms"].head(texts)
        model, *variant = run.approach.split(":")
        row = {
            "model": model,
            "variant": ":".join(variant),
            "task": run.task,
            "n": run.n,
            "p50_ms": first.median(),
            "p95_ms": first.quantile(0.95),
            "peak_mb": run.peak_mb,
            "macro_f1": run.macro_f1 if run.table == "results" else None,
        }
        if run.table == "results" and variant[0] in ("q4_0", "q8_0"):
            mlx = runs[(runs["approach"] == f"{model}:{variant[0][1]}bit") & (runs["task"] == run.task)]
            if len(mlx):
                other = RESULTS / "results-runs" / run.task / mlx["run_id"].iloc[0] / "predictions.csv"
                row["agree_mlx"] = (predictions["predicted"] == pd.read_csv(other)["predicted"]).mean()
        rows.append(row)
    table = pd.DataFrame(rows).drop_duplicates(["model", "variant", "task", "n"], keep="last")
    print(f"\n4B models, MLX and llama.cpp: latency over the first {texts} test texts\n", sep="")
    for task in TASKS:
        part = table[table["task"] == task].drop(columns="task").sort_values(["model", "variant", "n"])
        print(f"{task}\n", part.round(4).to_string(index=False), "\n", sep="")


if __name__ == "__main__":
    sweep()
    tables()
    runtimes()
