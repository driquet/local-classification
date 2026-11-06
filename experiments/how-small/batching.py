"""Does batching change where to run an encoder? Texts per second at batch sizes 1 to 128.

The rest of the experiment classifies one text at a time, like a service answering requests. A backlog can
be classified in batches instead. This probe measures throughput on the first 1,000 test texts of CLINC150
and VAST, sorted by length so that each batch holds texts of similar length, as a batch job would do.
Each setup and batch size runs in a fresh process, so its peak memory is its own. The encoders are the
"full" stage's seed 0 checkpoints: throughput depends on the architecture, not on the training.

    uv run experiments/how-small/batching.py        # alone on the machine
    uv run experiments/how-small/batching.py 40     # smoke test on 40 texts, nothing written
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

from lc import data
from lc.measure import in_fresh_process

HERE = Path(__file__).resolve().parent
CHECKPOINTS = HERE / "results" / "checkpoints"
OUT = HERE / "results" / "batching.csv"
GTE = ("thenlper/gte-base", "c078288308d8dee004ab72c6191778064285ec0c")
TASKS = ["clinc150", "vast"]
BATCHES = [1, 8, 32, 128]
MAX_LENGTH = 512
SETUPS = [
    ("ettin-17m", "int8, CPU"),
    ("ettin-150m", "int8, CPU"),
    ("ettin-17m", "fp16, GPU"),
    ("ettin-150m", "fp16, GPU"),
    ("gte-base", "fp32, GPU"),
]


def by_length(texts: list[str], tokenizer, size: int) -> list[list[str]]:
    order = sorted(range(len(texts)), key=lambda i: len(tokenizer(texts[i], truncation=True)["input_ids"]))
    return [[texts[i] for i in order[k : k + size]] for k in range(0, len(order), size)]


def throughput(model: str, setup: str, task: str, size: int, n_texts: int) -> float:
    """Texts per second over ``n_texts`` test texts, in batches of ``size``. Runs in a fresh process."""
    from transformers import AutoTokenizer

    texts = data.load(task, "test")["text"].tolist()[:n_texts]
    checkpoint = CHECKPOINTS / model / task / "seed-0"
    if setup == "int8, CPU":
        import onnxruntime as ort

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 1, 1
        session = ort.InferenceSession(
            checkpoint.with_name("seed-0-int8") / "model.int8.onnx",
            options,
            providers=["CPUExecutionProvider"],
        )

        def run(batch: list[str]) -> None:
            encoded = tokenizer(
                batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="np"
            )
            inputs = {name: encoded[name].astype(np.int64) for name in ("input_ids", "attention_mask")}
            session.run(None, inputs)

    else:
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification

        if model == "gte-base":
            tokenizer = AutoTokenizer.from_pretrained(GTE[0], revision=GTE[1])
            net = AutoModel.from_pretrained(GTE[0], revision=GTE[1])
        else:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            net = AutoModelForSequenceClassification.from_pretrained(checkpoint, dtype=torch.float16)
        net = net.to("mps").eval()

        @torch.inference_mode()
        def run(batch: list[str]) -> None:
            encoded = tokenizer(
                batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
            )
            out = net(**encoded.to("mps"))
            scores = out.logits if hasattr(out, "logits") else out.last_hidden_state.mean(1)
            scores.cpu()  # waits for the GPU

    batches = by_length(texts, tokenizer, size)
    run(batches[len(batches) // 2])  # warm-up
    start = time.perf_counter()
    for batch in batches:
        run(batch)
    return len(texts) / (time.perf_counter() - start)


def main() -> None:
    n_texts = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rows = []
    for model, setup in SETUPS:
        for task in TASKS:
            for size in BATCHES:
                rate, peak_mb = in_fresh_process(throughput, model, setup, task, size, n_texts)
                rows.append(
                    {
                        "model": model,
                        "setup": setup,
                        "task": task,
                        "batch": size,
                        "texts_per_s": round(rate, 1),
                        "ms_per_text": round(1000 / rate, 3),
                        "peak_mb": peak_mb,
                    }
                )
                print(rows[-1], flush=True)
    if len(sys.argv) > 1:
        return
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
