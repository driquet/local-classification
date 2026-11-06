"""Is a 4B model slow because of the number of labels, or because of the length of the prompt?

On CLINC150 the zero-shot prompt lists every intent, so the two grow together. This probe separates them
on the texts of the 10-intent slice, with Qwen3.5-4B in 8 bits on the GPU, in a 2x2:

- prompt: the 10 intents, or long, padded with neutral text to the length of the 150-intent prompt
- scored: the 10 intents, or all 150 (the extra ones are scored without being listed: a latency probe only)

The plain 150-intent prompt is a fifth cell, for reference. Run it alone on the machine.

    uv run experiments/how-small/prompt_length.py
"""

import csv
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from lc import data
from lc.models.prompted import LabelScorer, readable, render

HERE = Path(__file__).resolve().parent
MODEL = HERE / "results" / "models" / "qwen3.5-4b-zero-shot-8bit"
TEMPLATE = (HERE.parent / "do-you-need-an-llm" / "prompts" / "clinc150" / "zero-shot.txt").read_text()
OUT = HERE / "results" / "prompt-length.csv"
TEXTS, WARMUP = 200, 5
FILLER = (
    "The messages come from users of a virtual assistant that helps with banking, travel, cooking, "
    "work, small talk and everyday tasks. "
)


def padded(template: str, labels: list[str], target_tokens: int, scorer: LabelScorer) -> str:
    """The template with filler before it, until its prompt is as long as ``target_tokens``."""
    filler = ""
    while len(scorer.prompt_ids(render(filler + template, "x", labels))) < target_tokens:
        filler += FILLER
    return filler + "\n" + template


def measure(scorer: LabelScorer, template: str, listed: list[str], scored: list[str], texts: list[str]):
    scorer.share_prefix(render(template, "a", listed), render(template, "b", listed))
    for text in texts[:WARMUP]:
        scorer.score(render(template, text, listed), scored)
    mx.clear_cache()
    mx.reset_peak_memory()
    times = []
    for text in texts:
        start = time.perf_counter()
        scorer.score(render(template, text, listed), scored)
        times.append((time.perf_counter() - start) * 1000)
    tokens = len(scorer.prompt_ids(render(template, texts[0], listed)))
    # Active memory is what the arrays hold; the cache is freed buffers MLX keeps for reuse.
    gb = 2**30
    return (
        tokens,
        np.percentile(times, 50),
        np.percentile(times, 95),
        mx.get_peak_memory() / gb,
        mx.get_cache_memory() / gb,
    )


def main() -> None:
    texts = data.load("clinc150-k10-c0", "test")["text"].tolist()[:TEXTS]
    ten = [readable(label) for label in data.labels("clinc150-k10-c0")]
    all_ = [readable(label) for label in data.labels("clinc150")]
    scorer = LabelScorer(str(MODEL), None)
    long_tokens = len(scorer.prompt_ids(render(TEMPLATE, "x", all_)))
    long_template = padded(TEMPLATE, ten, long_tokens, scorer)
    cells = [
        ("10 intents", TEMPLATE, ten, ten),
        ("10 intents, padded", long_template, ten, ten),
        ("10 intents", TEMPLATE, ten, all_),
        ("10 intents, padded", long_template, ten, all_),
        ("150 intents", TEMPLATE, all_, all_),
    ]
    rows = []
    for name, template, listed, scored in cells:
        tokens, p50, p95, peak, cache = measure(scorer, template, listed, scored, texts)
        rows.append(
            {
                "prompt": name,
                "labels_scored": len(scored),
                "prompt_tokens": tokens,
                "p50_ms": round(p50, 1),
                "p95_ms": round(p95, 1),
                "peak_active_gb": round(peak, 2),
                "cache_gb": round(cache, 2),
            }
        )
        print(rows[-1], flush=True)
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
