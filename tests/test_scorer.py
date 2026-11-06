"""The label scorer on a small model (Qwen3.5-0.8B, downloaded on first run). Uses the GPU."""

import mlx.core as mx
import numpy as np
import pytest

from lc.models import prompted
from lc.models.prompted import LabelScorer

pytestmark = pytest.mark.gpu

MODEL = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
LABELS = ["card arrival", "card linking", "lost or stolen card", "refund not showing up", "age limit"]
TEMPLATE = (
    "Classify the message into one of: " + ", ".join(LABELS) + ".\n\nMessage: {}\n\nAnswer with the label."
)
TEXTS = ["I still have not received my new card.", "How old do I need to be to open an account?"]
# The tests run the model in float32. In its native bfloat16, a different batch shape or order of operations
# alone moves probabilities by up to about 0.01, which would hide a real error of that size.
ATOL = 5e-3


@pytest.fixture(scope="module")
def scorer():
    scorer = LabelScorer(*MODEL)
    scorer.model.set_dtype(mx.float32)
    return scorer


def test_probabilities_sum_to_one(scorer):
    for text in TEXTS:
        probs = scorer.score(TEMPLATE.format(text), LABELS, use_cache=False)
        assert probs.shape == (len(LABELS),) and probs.sum() == pytest.approx(1.0)


def test_shared_prefix_scoring_agrees_with_uncached_scoring(scorer):
    assert scorer.share_prefix(*(TEMPLATE.format(text) for text in TEXTS)) > 20
    # The last text's leading newline merges with the prefix's last token: it is encoded in full instead.
    for text in [*TEXTS, "\n leading space"]:
        cached = scorer.score(TEMPLATE.format(text), LABELS)
        uncached = scorer.score(TEMPLATE.format(text), LABELS, use_cache=False)
        np.testing.assert_allclose(cached, uncached, atol=ATOL)


def test_labels_scored_in_chunks_agree_with_one_pass(scorer, monkeypatch):
    message = TEMPLATE.format(TEXTS[0])
    one_pass = scorer.score(message, LABELS)
    monkeypatch.setattr(prompted, "LABEL_CHUNK", 2)
    np.testing.assert_allclose(scorer.score(message, LABELS), one_pass, atol=ATOL)


def test_shuffling_labels_permutes_probabilities(scorer):
    message = TEMPLATE.format(TEXTS[1])
    order = [3, 0, 4, 1, 2]
    probs = scorer.score(message, LABELS)
    shuffled = scorer.score(message, [LABELS[i] for i in order])
    np.testing.assert_allclose(shuffled, probs[order], atol=ATOL)
