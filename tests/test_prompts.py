import functools
import json
from pathlib import Path

import pytest

from lc import data
from lc.data import CLINC150
from lc.models.prompted import PromptedLLM, readable, render

PROMPTS = Path(__file__).resolve().parents[1] / "experiments" / "do-you-need-an-llm" / "prompts"
QWEN = ("Qwen/Qwen3.5-4B", "rev")


@pytest.fixture(autouse=True)
def toy_examples(data_dir, monkeypatch):
    """Few-shot examples drawn from the toy tasks instead of the real data directory."""
    monkeypatch.setattr(
        data, "few_shot_examples", functools.partial(data.few_shot_examples, data_dir=data_dir)
    )


def rendered(classifier: PromptedLLM, task: str, data_dir, tmp_path, texts):
    """The prompts the predictor would score, as rendered from what ``fit`` writes."""
    classifier.fit(task, data.load(task, "train", data_dir), data.labels(task, data_dir), 0, tmp_path)
    spec = json.loads((tmp_path / "prompt.json").read_text())
    return [render(spec["template"], text, spec["labels"], spec["examples"]) for text in texts], spec


def test_few_shot_examples_come_before_the_text_and_appear_once(data_dir, tmp_path):
    few_shot = PromptedLLM(*QWEN, str(PROMPTS), shots=(("vast", 5),))
    texts = ["Topic: a\nComment: first query.", "Topic: b\nComment: second query."]
    prompts, spec = rendered(few_shot, "vast", data_dir, tmp_path, texts)
    shared = prompts[0].split(texts[0])[0]
    assert prompts[1].startswith(shared) and "Stance: against" in shared
    examples = spec["examples"].split("\n\n")
    assert len(examples) == 15
    assert all(prompts[0].count(example) == 1 for example in examples)


def test_dynasent_prompts_live_apart_and_show_five_examples_per_sentiment(data_dir, tmp_path):
    few_shot = PromptedLLM(*QWEN, str(PROMPTS.with_name("prompts-dynasent")), shots=(("dynasent", 5),))
    text = "The art exhibit has a lot to offer."
    prompts, spec = rendered(few_shot, "dynasent", data_dir, tmp_path, [text])
    assert len(spec["examples"].split("\n\n")) == 15 and "Sentiment: neutral" in spec["examples"]
    assert prompts[0].endswith(f"Sentence: {text}\n")
    assert not (PROMPTS / "dynasent").exists(), "would change the run ids of the CLINC150 and VAST runs"


def test_zero_shot_prompt_has_no_examples(data_dir, tmp_path):
    zero_shot = PromptedLLM(*QWEN, str(PROMPTS))
    prompts, spec = rendered(zero_shot, CLINC150, data_dir, tmp_path, ["wake me up at noon"])
    assert spec["examples"] == "" and "Examples:" not in prompts[0]
    assert "out of scope" in prompts[0] and "intent 000" in prompts[0]


def test_labels_read_as_words():
    assert readable("pto_request") == "pto request"
    assert readable("oos") == "out of scope"
