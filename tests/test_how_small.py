"""The plan of the how-small experiment, and the serving variants of one checkpoint."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lc.models.tfidf import TfidfLogisticRegression
from lc.runs import Run, execute

RUN_PY = Path(__file__).parents[1] / "experiments" / "how-small" / "run.py"
FIRST_POST = RUN_PY.parents[1] / "do-you-need-an-llm" / "results" / "results.csv"
TEXTS = ["what is my credit limit", "book a table for two at eight tonight", "bad " * 700]


@pytest.fixture(scope="module")
def how_small():
    spec = importlib.util.spec_from_file_location("how_small_run", RUN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_checkpoint_outlives_its_run(data_dir, tmp_path):
    kept = tmp_path / "kept"
    run = Run("tfidf", "vast", per_class=10)
    execute(
        run,
        TfidfLogisticRegression(),
        tmp_path / "r.csv",
        tmp_path / "runs",
        data_dir=data_dir,
        checkpoint=kept,
    )
    assert (kept / "model.pkl").exists()


def test_every_job_is_unique_and_named(how_small):
    jobs = how_small.plan(how_small.STAGES, how_small.TASKS)
    assert len({(job.run, job.table) for job in jobs}) == len(jobs)
    recipe = {"learning_rate": 1e-4, "epochs": 4}
    how_small.RECIPES.update({model: recipe for model in [*how_small.SMALL, *how_small.TUNED]})
    for job in jobs:
        if job.run.approach.split(":")[0] not in how_small.LLMS:
            how_small.classifier(job.run.approach, job.run.task, Path("checkpoints"))


def test_the_sweep_winner_has_the_best_mean_over_both_tasks(how_small, tmp_path, monkeypatch):
    scores = {"3e-5": (0.90, 0.70), "5e-5": (0.95, 0.60), "1e-4": (0.80, 0.70)}
    rows = [
        {"approach": f"ettin-150m-4ep:lr{rate}", "task": task, "macro_f1": f1}
        for rate, pair in scores.items()
        for task, f1 in zip(["clinc150", "vast"], pair, strict=True)
    ]
    monkeypatch.setattr(how_small, "RESULTS_DIR", tmp_path)
    pd.DataFrame(rows[:-1]).to_csv(tmp_path / "sweep.csv", index=False)
    with pytest.raises(SystemExit):
        how_small.swept("ettin-150m-4ep")
    pd.DataFrame(rows).to_csv(tmp_path / "sweep.csv", index=False)
    assert how_small.swept("ettin-150m-4ep") == {"learning_rate": 3e-5, "epochs": 4}


def test_only_filters_by_prefix(how_small):
    jobs = how_small.plan(["serve"], ["vast"], ["ettin-17m"])
    assert jobs and all(job.run.approach.startswith("ettin-17m:") for job in jobs)


MLX_8BIT = RUN_PY.parent / "results" / "models" / "onejev1.6-4b-8bit" / "config.json"


@pytest.mark.skipif(not MLX_8BIT.exists(), reason="needs the 8-bit weights of the llm stage")
def test_a_4b_model_on_the_cpu_is_routed_by_its_device(how_small):
    from lc.models.gguf import GgufDecisionModel
    from lc.models.prompted import OnCpu
    from lc.runs import run_id

    gpu = how_small.classifier("onejev1.6-4b:8bit", "dynasent", Path("checkpoints"))
    cpu = how_small.classifier("onejev1.6-4b:8bit:cpu", "dynasent", Path("checkpoints"))
    assert isinstance(cpu, OnCpu) and cpu.inner == gpu
    assert cpu.identity() == {**gpu.identity(), "device": "cpu"}
    # The GPU run keeps its id when a CPU variant is added, so it is not run again.
    results = pd.read_csv(RUN_PY.parent / "results" / "results.csv")
    assert run_id(Run("onejev1.6-4b:8bit", "dynasent"), gpu) in set(results["run_id"])
    bf16 = how_small.classifier("onejev1.6-4b:bf16:cpu", "dynasent", Path("checkpoints"))
    assert bf16.inner.repo == how_small.LLMS["onejev1.6-4b"][0]
    gguf = how_small.classifier("onejev1.6-4b:bf16:cpu4", "dynasent", Path("checkpoints"))
    assert isinstance(gguf, GgufDecisionModel) and gguf.gguf.endswith("-BF16.gguf")


@pytest.mark.skipif(not FIRST_POST.exists(), reason="needs the first post's results")
def test_serving_ettin_150m_keeps_the_first_posts_run_ids(how_small):
    from lc.runs import run_id

    how_small.RECIPES["ettin-150m"] = {}
    first = pd.read_csv(FIRST_POST)
    run = Run("ettin-150m", "vast", 100, 1)
    assert run_id(run, how_small.fine_tuned("ettin-150m")) in set(first["run_id"])


@pytest.mark.gpu
def test_int8_and_half_precision_serve_a_checkpoint(tmp_path):
    """Same tokens as transformers and a quarter of the size; half precision stays close.

    How far int8 drifts is what the experiment measures, on trained heads: here it only has to run.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from lc.models.encoder import EncoderPredictor
    from lc.models.onnx import LABELS_FILE, OnnxPredictor, export_int8

    repo, revision = "jhu-clsp/ettin-encoder-17m", "987607455c61e7a5bbc85f7758e0512ea6d0ae4c"
    AutoTokenizer.from_pretrained(repo, revision=revision).save_pretrained(tmp_path)
    AutoModelForSequenceClassification.from_pretrained(repo, revision=revision, num_labels=3).save_pretrained(
        tmp_path
    )
    (tmp_path / LABELS_FILE).write_text('["a", "b", "c"]')
    export_int8(tmp_path, tmp_path / "int8")
    (tmp_path / "int8" / "tokenizer.json").write_bytes((tmp_path / "tokenizer.json").read_bytes())

    onnx = OnnxPredictor(tmp_path / "int8", 512, 1)
    fp32 = EncoderPredictor(tmp_path, 512, "cpu")
    assert (tmp_path / "int8" / "model.int8.onnx").stat().st_size < 0.3 * (
        tmp_path / "model.safetensors"
    ).stat().st_size
    for text in TEXTS:
        expected = fp32.tokenizer([text], truncation=True, max_length=512)["input_ids"][0]
        assert onnx.tokenizer.encode(text).ids == expected
        assert abs(onnx.predict(text).sum() - 1) < 1e-6

    half = EncoderPredictor(tmp_path, 512, "mps", half=True)
    assert np.abs(half.predict(TEXTS[0]) - fp32.predict(TEXTS[0])).max() < 0.1


def test_a_table_from_before_a_new_column_keeps_its_rows(tmp_path):
    from lc.runs import COLUMNS, append_row

    path = tmp_path / "results.csv"
    old = [column for column in COLUMNS if column != "gpu_used_mb"]
    pd.DataFrame([{**dict.fromkeys(old, "x"), "run_id": "0123"}]).to_csv(path, index=False)
    append_row({**dict.fromkeys(COLUMNS, "y"), "run_id": "4567"}, path)
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(table.columns) == COLUMNS
    assert table["run_id"].tolist() == ["0123", "4567"]  # leading zeros survive the rewrite
    assert table["gpu_used_mb"].tolist() == ["", "y"]


def test_amber_maps_its_options_back_to_the_labels(how_small):
    from lc.models.amber import AmberDecisionModel, AmberPredictor

    cpu = how_small.classifier("amber-150m:fp32:cpu1", "vast", Path("checkpoints"))
    assert cpu == AmberDecisionModel(*how_small.AMBER["amber-150m"], device="cpu", threads=1, dtype="fp32")
    assert how_small.classifier("amber-32m:bf16:mps", "vast", Path("checkpoints")).device == "mps"
    assert how_small.classifier("amber-32m:int8:cpu4", "vast", Path("checkpoints")).threads == 4

    class Engine:  # VAST's options are A: against, B: favor, C: neutral
        def __call__(self, state, questions):
            assert questions["q"]["criteria"] == {"A": "against", "B": "favor", "C": "neutral"}
            return {"answers": {"q": {"probabilities": {"A": 0.1, "B": 0.7, "C": 0.2}}}}, None

    probs = AmberPredictor(Engine(), "vast", ["for", "against", "neutral"]).predict("Topic: x\nComment: y")
    assert probs.tolist() == [0.7, 0.1, 0.2]
