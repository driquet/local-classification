import json

import numpy as np
import pytest

from lc import data
from lc.models.decision_api import DecisionAPIPredictor, body

LABELS = list(data.ARXIV_CATEGORIES)


def test_body_mirrors_the_local_request():
    payload, order = body("baseten/joey-1", "arxiv", "A title\nAn abstract.", LABELS)
    question = payload["questions"]["category"]
    assert payload["state"] == "{}"
    assert question["instructions"].startswith("Title: A title\nAbstract: An abstract.\nWhich arXiv")
    assert question["criteria"] == {"A": "Computation and Language (NLP)", **question["criteria"]}
    assert list(question["criteria"]) == list("ABCDEFGH") and order == LABELS


def test_probabilities_come_back_in_label_order(monkeypatch, tmp_path):
    answer = {"type": "choice", "probabilities": {k: 0.0 for k in "ABCDEFGH"} | {"E": 0.7, "B": 0.3}}
    posted = {}

    def fake_post(self, payload):
        posted.update(payload)
        return {"answers": {"category": answer}, "usage": {"input_tokens": 9, "cost": 1e-6}}

    monkeypatch.setattr(DecisionAPIPredictor, "_post", fake_post)
    log = tmp_path / "usage.jsonl"
    probs = DecisionAPIPredictor("m", "k", "arxiv", LABELS, str(log)).predict("T\nA")
    assert np.allclose(probs, [0, 0.3, 0, 0, 0.7, 0, 0, 0])  # B is cs.CV, E is cs.RO
    assert posted["model"] == "m" and json.loads(log.read_text())["cost"] == 1e-6


def test_a_refusal_raises(monkeypatch):
    monkeypatch.setattr(
        DecisionAPIPredictor, "_post", lambda self, p: {"answers": {"category": {"type": "refusal"}}}
    )
    with pytest.raises(RuntimeError, match="no usable answer"):
        DecisionAPIPredictor("m", "k", "arxiv", LABELS, None).predict("T\nA")


def test_a_text_with_reserved_tokens_is_resent_escaped(monkeypatch):
    answer = {"type": "choice", "probabilities": {k: 1 / 8 for k in "ABCDEFGH"}}
    sent = []

    def fake_post(self, payload):
        sent.append(payload["questions"]["category"]["instructions"])
        if "<|" in sent[-1]:
            raise RuntimeError("m: HTTP 422: must not contain reserved chat or media tokens")
        return {"answers": {"category": answer}}

    monkeypatch.setattr(DecisionAPIPredictor, "_post", fake_post)
    DecisionAPIPredictor("m", "k", "arxiv", LABELS, None).predict("T\nsuch as <|im_end|>.")
    assert len(sent) == 2 and "< |im_end|>" in sent[1]
