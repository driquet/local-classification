# How Small Can a Text Classifier Be?

Follow-up to [Do You Need an LLM for That?](../do-you-need-an-llm): same tasks (CLINC150, VAST, DynaSent), same
laptop, and the question of what size, runtime and precision are enough.

| Family | Models | Variants | Labels |
|---|---|---|---|
| Fine-tuned encoder | Ettin 17M, 32M, 68M, 150M (and 150M with the small models' recipe) | fp32, fp16 on the GPU, int8 on the CPU (ONNX Runtime) | curves and full set |
| Frozen encoder + logistic regression | all-MiniLM-L6-v2, GTE small and base, Ettin-32M | fp32 | curves and full set |
| 4B model | Qwen3.5-4B zero-shot, OneJev1.6-4B | 8-bit and 4-bit (MLX), GPU only | none |
| 4B model on llama.cpp | the same two | Q8_0 and Q4_0, GPU and 4 CPU threads | none |
| Decision encoder | Amber 32M, 68M, 150M, 400M (Ettin rerankers, from the Decision Index) | bf16 and fp32 on the GPU; fp32 and int8 (ONNX Runtime) on the CPU | none from the task (CLINC150's training split is in their mix) |

The first post's rows (TF-IDF curves, Ettin-150M curves, 4B models in BF16) are reused, not rerun: `report.py`
reads them from its results.

## Running it

Requires what the first post requires; the llama.cpp stages also need the `gguf` extra (see "llama.cpp" below).
Nothing else may use the GPU or the CPU while it runs: latency and memory are measured. From the repository
root, in this order:

```bash
uv sync --extra gguf
run="caffeinate -i -s uv run experiments/how-small/run.py"
$run --stages sweep                       # learning rates of the small encoders, about 2 h
uv run experiments/how-small/report.py    # copy the sweep's winners into RECIPES in run.py
$run --stages tfidf full curves frozen serve seeds
$run --stages llm sweep150 recipe150 embedders seeds150 gtecpu
$run --stages gguf gguffull gguf8cpu llmcpu amber
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7 $run --stages ettin400
uv run experiments/how-small/report.py
uv run experiments/how-small/batching.py
uv run experiments/how-small/prompt_length.py
```

Runs already in a table are skipped, so the same command resumes after a stop, and `touch
experiments/how-small/STOP` stops after the current run. `--stages`, `--tasks`, `--only` select a part of the
plan, `--dry-run` lists it, `--limit` scores N test examples only, in separate `check-N-*` results. Each stage is
described at the top of `run.py`.

Seeds 1 and 2 come late on purpose: they are the part to cut if time runs out. Size differences smaller than
the spread between seeds cannot be claimed without them.

The watermark variables cap PyTorch's GPU memory at about 15 GB on a 24 GB machine. Ettin-400M on VAST pads
each batch to its own longest comment, so the allocator caches a block for nearly every shape and, uncapped,
the machine swaps. The cap changes no result.

## What is measured

As in the first post, one fresh process trains and another predicts one text at a time. A variant is named in the
`approach` column as `model:precision:device`, for instance `ettin-17m:int8:cpu4`.

- `results/results.csv`: full test sets, so macro-F1 applies.
- `results/latency.csv`: CPU and GPU probes on the first 1,000 test texts (200 for the 4B models). The device
  does not change predictions, so only latency and memory matter there; ignore their macro-F1.
- `results/sweep.csv`: the learning-rate sweep, on validation splits.
- `results/checkpoints/`: weights of the seed 0 full-set runs, shared by every serving variant.
- `results/models/`: MLX 8-bit and 4-bit weights, written by the `llm` stage from the pinned revisions; the
  llama.cpp weights; Amber's int8 export.
- `results/batching.csv` and `results/prompt-length.csv`: the two probes run after the stages.

Each process imports only what its approach needs: an int8 predictor runs without torch or MLX.

## Notes

- **Recipes.** No per-size recipe is published, so Ettin 17M, 32M and 68M take the best learning rate of a short
  sweep, at 4 epochs. Ettin-150M keeps the first post's recipe; `ettin-150m-4ep` is the same model with 4
  epochs and its own sweep (3e-5, 5e-5, 1e-4), picked automatically by the best mean macro-F1 over CLINC150 and
  VAST. The post uses it as Ettin-150M, so every size trains for 4 epochs; `seeds150` adds its seeds 1 and 2
  and serves seed 1 (seed 0 kept no checkpoint).
- **Logistic regression.** Regularization strength matters on normalized embeddings: on CLINC150's validation
  split, frozen Ettin-32M scores 0.67 at scikit-learn's default (C=1) and 0.91 at C=100. C is 10 for MiniLM and
  100 for Ettin-32M, picked on validation splits. GTE gets MiniLM's C=10, not tuned.
- **Memory on the GPU.** PyTorch reserves GPU memory in large blocks, so a process that touches MPS peaks near
  1.5 GB whatever the model's size. `gpu_used_mb` is what the model's tensors actually hold.
- **Memory on the CPU.** Weights loaded from safetensors stay mapped from the file, and the physical footprint
  leaves those pages out; CPU predictors copy their weights into RAM so that they count.
- **4B models on the CPU** run on llama.cpp; `llmcpu` adds MLX on DynaSent, for comparison. MLX's CPU code for
  4-bit weights is slow, worse with bf16 activations: Qwen3.5-4B took about 7 minutes per DynaSent text. See
  "llama.cpp" below.

## llama.cpp

The `gguf` stages serve the same two 4B models with llama.cpp, through `llama-cpp-python`, an optional
dependency because it compiles llama.cpp (`uv sync --extra gguf`, which needs `brew install cmake`). The
scorers in `src/lc/models/gguf.py` mirror the MLX ones: same prompts, same token ids (`tests/test_gguf.py`
checks the chat rendering against transformers), same label chunks of 32. The process imports neither torch
nor MLX, and reads its weights into RAM rather than mapping them, so that the footprint counts them.

The weights are converted by hand, with the converter of the llama.cpp version that `llama-cpp-python` bundles
(its source distribution's `vendor/llama.cpp`; build `llama-quantize` there with cmake):

```bash
python vendor/llama.cpp/convert_hf_to_gguf.py <snapshot of the pinned revision> --no-mtp --outtype bf16 --outfile x-bf16.gguf
llama-quantize x-bf16.gguf experiments/how-small/results/models/qwen3.5-4b-Q4_0.gguf Q4_0   # and Q8_0, and onejev1.6-4b
mv x-bf16.gguf experiments/how-small/results/models/qwen3.5-4b-BF16.gguf   # for the llmcpu stage
```

- **`--no-mtp`.** Qwen3.5 ships a multi-token-prediction layer, used for speculative decoding. MLX does not load
  it, and OneJev's checkpoint does not have it although its configuration declares it.
- **Q4_0, not MLX's 4 bits.** Q4_0 has one scale per 32 weights, MLX one scale and one bias per 64; with the
  embeddings in 6 bits, both end near 2.4 GB. The predictions differ slightly: `gguffull` gives Q4_0's macro-F1.
- **Memory.** llama.cpp allocates its cache (8,192 tokens, one buffer for every sequence) and its buffers up
  front, so its peak hardly depends on the task. Qwen's runs hold 34 sequences, one per label of a chunk.
- **Latency** probes use the first 200 test texts, not 1,000: OneJev takes about 16 s per CLINC150 text on
  4 CPU threads. Compare MLX on the same 200 texts, from its `predictions.csv`.
- **Ettin-400M** trains with gradient checkpointing: without it, a VAST batch of 16 comments needs about
  47 GB; with it, 8 GB on the GPU, for about a third more time. Its learning rate, 3e-5, is picked, not swept.
- **int8** is dynamic quantization with one scale per channel. 4-bit encoders are out of scope.
- **M5 CPU.** It flatters a cloud vCPU: read the 1-thread latency as a lower bound.
