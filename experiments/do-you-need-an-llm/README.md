# Do You Need an LLM for That? Classifying Text Locally

This folder reproduces every number in the post: five approaches, three datasets, one laptop.

| Approach | Labels it needs | Model |
|---|---|---|
| TF-IDF + logistic regression | the training set | scikit-learn |
| Fine-tuned encoder | the training set | [Ettin-150M](https://huggingface.co/jhu-clsp/ettin-encoder-150m) |
| Prompted LLM, zero-shot | none | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Prompted LLM, few-shot | 1 per intent, 5 per stance or sentiment | Qwen3.5-4B |
| Decision model | none | [OneJev1.6-4B](https://huggingface.co/OmniJev/OneJev1.6-4B) |

## Tasks

- **CLINC150**: intents of requests to a voice assistant, 150 intents plus out-of-scope, scored on the
  official test set (5,500 examples) as the [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)
  does.
- **CLINC150 slices**: the same data restricted to 10, 25, 50 and 100 intents, without out-of-scope. Every
  intent has 100 training and 30 test examples, so only the number of classes changes. At 10 and 25
  intents, which intents are drawn matters, so there are three random draws (chains).
- **VAST**: stance of a New York Times comment toward a topic (for, against, neutral), official splits.
- **DynaSent**: sentiment of a sentence (positive, negative, neutral), round 2, official splits.

The two trained approaches also run on smaller training sets (learning curves): 10, 25 and 50 examples per
intent on CLINC150, and 10 to 1,000 per class on VAST and DynaSent, with three random draws up to 100 per class.

## Running it

Requires macOS on Apple silicon with at least 24 GB of memory, and [uv](https://docs.astral.sh/uv/).
From the repository root:

```bash
uv sync
uv run experiments/do-you-need-an-llm/run.py --limit 50    # end-to-end check on 50 test examples per run
caffeinate -i -s uv run experiments/do-you-need-an-llm/run.py
uv run experiments/do-you-need-an-llm/report.py
```

`run.py` runs the 129 runs in order, cheapest first, and appends one row per run to
`results/results.csv`. Runs already in the table are skipped, so the same command resumes after a stop;
`touch experiments/do-you-need-an-llm/STOP` stops it cleanly after the current run. `--only` and `--tasks`
select a part of the plan, `--dry-run` lists it. `caffeinate` keeps the Mac awake (keep it on power, lid
open).

Measured on an Apple M5 with 24 GB:

| Block | Runs | Time |
|---|---|---|
| TF-IDF | 48 | 5 min |
| Ettin-150M | 48 | ~5 h |
| Qwen3.5-4B, zero-shot | 11 | ~3 h |
| Qwen3.5-4B, few-shot | 11 | ~4 h |
| OneJev1.6-4B | 11 | ~3.5 h |

Close other GPU-heavy applications while it runs: latency and memory are measured, and the 4B models need
most of the 24 GB.

## What is measured

Every run trains (or prepares) its model in a fresh process, then predicts the test set one text at a time
in another fresh process, as a service answering requests would. Each row of `results/results.csv` holds:

- **Quality**: macro-F1 and accuracy, with 95% bootstrap confidence intervals, and the F1 of every class.
- **Labeled data**: examples used for training, or shown in the prompt for few-shot.
- **Training**: wall time and the training process's peak memory.
- **Prediction**: median and 95th-percentile latency per text, and the peak memory of the prediction
  process, which is what serving the model takes.
- **Disk**: the trained model, or the model files for the LLMs.

`run.py` also writes the predictions of every run to `results/runs/<task>/<run id>/predictions.csv`, one
probability per label for every test example. `results/` is not in the repository: a fresh clone starts
empty, and `report.py` needs a finished `results/results.csv`.

## Notes

- **One seed.** Each full-training run and the few-shot draw happen once. The confidence intervals cover
  test-set noise, not training noise. Few-shot results in particular move with the examples drawn, most of
  all on VAST, where 15 examples carry everything the model sees.
- **Contamination.** CLINC150 and VAST have been public for years and are likely in Qwen's training data.
  OneJev1.6 was trained on "the training splits of public decision datasets", likely these two among them.
- **OneJev's interface.** Requests are rendered as the model's own serving code renders them and worded as
  the Decision Index words them, so the scores line up with the index's: 0.9578 on CLINC150 and 0.8068 on
  VAST (edition 0.4.1). DynaSent is not in the index, so its requests are worded after the VAST ones.
- **Licences.** CLINC150 is CC-BY-3.0, DynaSent CC-BY-4.0. VAST states no licence, so its splits are
  downloaded from the authors' repository rather than redistributed.
