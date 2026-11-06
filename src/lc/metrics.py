"""Quality metrics: macro-F1 and accuracy, with percentile-bootstrap confidence intervals.

Macro-F1 averages the F1 of every class, so a rare class weighs as much as a frequent one. It is also the
metric the Decision Index reports for CLINC150 and VAST, which puts every score on the same scale.
"""

import numpy as np

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0


def per_class_f1(gold: np.ndarray, pred: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """F1 of every class over integer labels, and a mask of the classes present in gold or predictions."""
    true_positives = np.bincount(gold[gold == pred], minlength=n_classes)
    support = np.bincount(gold, minlength=n_classes) + np.bincount(pred, minlength=n_classes)
    f1 = np.divide(2 * true_positives, support, out=np.zeros(n_classes), where=support > 0)
    return f1, support > 0


def macro_f1(gold: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """Mean F1 over the classes present in gold or predictions (scikit-learn's default)."""
    f1, present = per_class_f1(gold, pred, n_classes)
    return float(f1[present].mean())


def scores(gold: np.ndarray, pred: np.ndarray, n_classes: int) -> dict[str, float]:
    """Macro-F1 and accuracy, each with a 95% percentile-bootstrap interval over test examples.

    The intervals capture test-set noise only: how much a score would move on another test set of the same
    size. They do not capture training noise, which only repeated training runs measure.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.empty((N_BOOTSTRAP, 2))
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(gold), len(gold))
        boot[b] = macro_f1(gold[idx], pred[idx], n_classes), (gold[idx] == pred[idx]).mean()
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    return {
        "macro_f1": macro_f1(gold, pred, n_classes),
        "macro_f1_low": low[0],
        "macro_f1_high": high[0],
        "accuracy": float((gold == pred).mean()),
        "accuracy_low": low[1],
        "accuracy_high": high[1],
    }
