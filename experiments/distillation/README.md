# No Labels? Borrow Them: Distilling a Classifier Locally

Third post, after [Do You Need an LLM for That?](../do-you-need-an-llm) and [How Small Can a Text Classifier
Be?](../how-small). A decision model (the teacher) labels unlabeled texts once. A small encoder (the student) is
fine-tuned on its answers. Both are graded on gold labels that nothing trained on.

## The task: arXiv categories, 2026

From a paper's title and abstract, predict its primary category among eight CS ones: cs.CL, cs.CV, cs.LG,
cs.CR, cs.RO, cs.SE, cs.HC, cs.IR. The papers were first submitted from March to October 2026 (arXiv ids `2603`
to `2610`). They come from [librarian-bots/arxiv-metadata-snapshot](https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot)
at a pinned revision, without withdrawn papers. The primary category is the first one in `categories`.

| Split | Per category | Total | Labels the models see |
|---|---|---|---|
| Training pool | 1,000 | 8,000 | the teacher's (gold only for the ceiling, and to measure the teacher) |
| Validation | 125 | 1,000 | the teacher's |
| Test | 250 | 2,000 | none: gold, for grading only |

## Who is who

| Role | Model | Labels |
|---|---|---|
| Teachers, rented | Base Labs Joey 1 (2B), Elephant 1 (9B), Mammoth 1 (27B); Perplexity Decider Flash v1.2 (4B), Decider v1.2 (27B), all on [OpenRouter's Decisions API](https://openrouter.ai/docs/guides/features/decisions) | none: the question, the title and abstract, the eight category names |
| Teacher, local | OneJev1.6-4B (bf16, MLX) | the same |
| Students | Ettin-150M, part 2's recipe (learning rate 5e-5, 4 epochs, 512 tokens, no checkpoint selection) | the teacher's: hard (cross-entropy on its top answer) or soft (KL divergence to its probabilities at T = 2, times T²) |
| Ceiling | the same Ettin-150M | the 8,000 gold labels |
| Floor | TF-IDF + logistic regression | gold, and each teacher's |

Students run each batch of 16 in 4 parts of 4, for the same update: whole batches of 512-token abstracts do not
fit the 15 GB MPS cap. Joey 1 is too weak to be worth a student.

| Students | Seeds |
|---|---|
| Decider 27B and Mammoth 27B, soft and hard; gold ceiling | 3 |
| OneJev1.6-4B, soft and hard | 2 |
| Elephant 9B and Decider Flash 4B, soft | 1 |

## Running it

The rented teachers need an `OPENROUTER_API_KEY`; labeling everything costs about $0.86. OneJev1.6-4B's Q4_0
weights for llama.cpp come from part 2: make them as [its README](../how-small) says. Nothing else may use the
GPU or the CPU while it runs, since latency and memory are measured. From the repository root:

```bash
uv sync --extra gguf
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7
uv run experiments/distillation/run.py --pilot          # the teachers on 100 papers per split, about 2 cents
caffeinate -i -s uv run experiments/distillation/run.py # everything, about 32 h
uv run experiments/distillation/report.py               # every number in the post
```

Runs already in a table are skipped, so the same command resumes after a stop, and `touch
experiments/distillation/STOP` stops after the current run. `--stages` and `--only` select part of the plan,
`--dry-run` lists it. The stages are described at the top of `run.py`. The watermark variables cap PyTorch's
GPU memory, as in part 2.

## What is measured

- `results/results.csv`: a teacher's rows on the `train` and `validation` splits are its labeling (`predict_s`
  is the labeling time, accuracy is the quality of its silver labels). The `test` rows grade everyone.
- `results/latency.csv`: OneJev1.6-4B served in Q4_0 with llama.cpp, on the first 200 test papers.
- `results/results-runs/<task>/<run id>/predictions.csv`: probabilities per paper, for agreement and for the
  teacher's mistakes the student copies or fixes.
- `results/api-usage.jsonl`: tokens, cost and latency of every API call.
- `data/arxiv-silver-<teacher>/`: the student's task, written from the teacher's predictions.
