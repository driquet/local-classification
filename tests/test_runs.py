import numpy as np
import pandas as pd
import pytest

from lc.models.tfidf import TfidfLogisticRegression
from lc.runs import COLUMNS, Run, execute, validate


def test_malformed_probabilities_are_rejected():
    good = np.array([[0.2, 0.8], [1.0, 0.0]])
    assert validate(good, 2, 2) is good
    for bad in (good[:1], np.array([[0.5, 0.6], [1.0, 0.0]]), np.array([[-0.1, 1.1], [1.0, 0.0]])):
        with pytest.raises(ValueError):
            validate(bad, 2, 2)


def test_a_run_writes_one_row_and_resumes(data_dir, tmp_path):
    results, runs_dir = tmp_path / "results.csv", tmp_path / "runs"
    run = Run("tfidf", "vast", per_class=10, seed=1)
    row = execute(run, TfidfLogisticRegression(), results, runs_dir, data_dir=data_dir)
    assert row is not None and row["train_n"] == 30 and row["n"] == 60
    table = pd.read_csv(results)
    assert list(table.columns) == COLUMNS and len(table) == 1
    predictions = pd.read_csv(runs_dir / "vast" / row["run_id"] / "predictions.csv")
    assert len(predictions) == 60 and list(predictions.columns[3:]) == ["p:against", "p:for", "p:neutral"]
    assert execute(run, TfidfLogisticRegression(), results, runs_dir, data_dir=data_dir) is None
    assert len(pd.read_csv(results)) == 1
