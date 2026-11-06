import numpy as np
import pytest

from lc.metrics import macro_f1, per_class_f1, scores


def test_perfect_predictions_score_one():
    gold = np.array([0, 1, 2, 2, 1, 0])
    result = scores(gold, gold.copy(), 3)
    assert result["macro_f1"] == 1.0 and result["accuracy"] == 1.0
    assert result["macro_f1_low"] == 1.0 and result["macro_f1_high"] == 1.0


def test_known_confusion_matrix():
    # Class 0: 2 right, 1 predicted as 1. Class 1: 1 right. Class 2: 1 right, 1 predicted as 0.
    gold = np.array([0, 0, 0, 1, 2, 2])
    pred = np.array([0, 0, 1, 1, 2, 0])
    f1, present = per_class_f1(gold, pred, 3)
    assert f1 == pytest.approx([2 * 2 / (3 + 3), 2 * 1 / (1 + 2), 2 * 1 / (2 + 1)])
    assert present.all()
    assert macro_f1(gold, pred, 3) == pytest.approx(f1.mean())


def test_absent_classes_do_not_count():
    gold = np.array([0, 0, 1, 1])
    assert macro_f1(gold, gold.copy(), 5) == 1.0


def test_interval_contains_the_point_estimate():
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 4, 400)
    pred = np.where(rng.random(400) < 0.7, gold, rng.integers(0, 4, 400))
    result = scores(gold, pred, 4)
    assert result["macro_f1_low"] < result["macro_f1"] < result["macro_f1_high"]
    assert result["accuracy_low"] < result["accuracy"] < result["accuracy_high"]
