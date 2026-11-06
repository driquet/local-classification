"""Turn the results table into the post's tables and chart data.

    uv run experiments/do-you-need-an-llm/report.py

Prints the tables in Markdown and writes the class-sweep chart series to ``results/class_sweep.json``.
"""

import json
from pathlib import Path

import pandas as pd

from lc.data import CLINC150

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "results.csv"

TRAINED = ["tfidf", "ettin-150m"]
ORDER = ["tfidf", "ettin-150m", "qwen3.5-4b-zero-shot", "qwen3.5-4b-few-shot", "onejev1.6-4b"]
NAMES = {
    "tfidf": "TF-IDF + LR",
    "ettin-150m": "Ettin-150M",
    "qwen3.5-4b-zero-shot": "Qwen3.5-4B, zero-shot",
    "qwen3.5-4b-few-shot": "Qwen3.5-4B, few-shot",
    "onejev1.6-4b": "OneJev1.6-4B",
}
# Decision Index 0.4.1 (data of 2026-10-29), https://huggingface.co/spaces/multimodalart/jev-decision-index
REFERENCES = {
    "Jev (Decision Index)": {CLINC150: 0.9028, "vast": 0.7365},
    "Best entrant (Decision Index)": {CLINC150: 0.976, "vast": 0.8418},
}


def class_sweep(full: pd.DataFrame) -> pd.DataFrame:
    """Macro-F1 by number of intents, averaged over the chains; the last column is the full task."""
    clinc = full[full["task"].str.startswith(CLINC150)].copy()
    clinc["intents"] = clinc["n_classes"].where(clinc["task"] != CLINC150, "150 + OOS").astype(str)
    table = clinc.pivot_table(index="approach", columns="intents", values="macro_f1", aggfunc="mean")
    return table[["10", "25", "50", "100", "150 + OOS"]].reindex(ORDER).dropna(how="all")


def three_classes(full: pd.DataFrame, task: str) -> pd.DataFrame:
    """VAST or DynaSent: labels used, macro-F1 and the F1 of the neutral class."""
    rows = full[full["task"] == task].set_index("approach").reindex(ORDER).dropna(how="all")
    neutral = rows["per_class_f1"].map(lambda f1: json.loads(f1)["neutral"])
    return pd.DataFrame(
        {"labels used": rows["train_n"].astype("Int64"), "macro-F1": rows["macro_f1"], "neutral F1": neutral}
    )


def learning_curves(results: pd.DataFrame) -> pd.DataFrame:
    """Mean macro-F1 over seeds by training examples per class, with the lowest and highest seed."""
    curves = results[results["approach"].isin(TRAINED) & results["task"].isin(["vast", "dynasent", CLINC150])]
    grouped = curves.groupby(["task", "approach", "per_class"])["macro_f1"]
    table = grouped.agg(["mean", "min", "max", "count"]).reset_index()
    table["size"] = pd.to_numeric(table["per_class"], errors="coerce").fillna(10**6)
    table = table.sort_values(["task", "approach", "size"]).drop(columns="size")
    return table.set_index(["task", "approach", "per_class"])


def cost(full: pd.DataFrame) -> pd.DataFrame:
    rows = full[full["task"] == CLINC150].set_index("approach").reindex(ORDER).dropna(how="all")
    return pd.DataFrame(
        {
            "macro-F1": rows["macro_f1"],
            "labels": rows["train_n"].astype("Int64"),
            "training (s)": rows["fit_s"].where(rows.index.isin(TRAINED)),
            "latency p50 (ms)": rows["latency_p50_ms"],
            "peak memory (GB)": (rows["peak_mb"] / 1024).round(2),
            "disk (MB)": rows["disk_mb"],
        }
    )


def markdown(table: pd.DataFrame) -> str:
    """A Markdown table, index first, empty cells for missing values."""
    frame = table.reset_index().rename(columns={"index": ""})
    header = [str(column) for column in frame.columns]
    rows = [["" if pd.isna(value) else str(value) for value in row] for row in frame.itertuples(index=False)]
    lines = [header, ["---"] * len(header), *rows]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines)


def main() -> None:
    results = pd.read_csv(RESULTS)
    full = results[results["per_class"] == "all"]
    for title, table in [
        ("Macro-F1 by number of intents (CLINC150)", class_sweep(full)),
        ("VAST", three_classes(full, "vast")),
        ("DynaSent, round 2", three_classes(full, "dynasent")),
        ("Learning curves (training examples per class)", learning_curves(results)),
        ("Cost on CLINC150 with out-of-scope", cost(full)),
        ("Decision Index references", pd.DataFrame(REFERENCES).T),
    ]:
        print(f"\n## {title}\n")
        print(markdown(table.rename(index=NAMES).round(4)))

    series = class_sweep(full).rename(index=NAMES)
    chart = {
        "labels": list(series.columns),
        "series": {name: list(row.round(3)) for name, row in series.iterrows()},
    }
    (HERE / "results" / "class_sweep.json").write_text(json.dumps(chart, indent=2) + "\n")


if __name__ == "__main__":
    main()
