import pytest

from lc import data
from lc.data import CLINC150, OUT_OF_SCOPE, class_slice, few_shot_examples, slice_name, subset

from .conftest import INTENTS, toy_splits


def test_slices_of_a_chain_are_nested_and_chains_differ():
    splits = toy_splits(INTENTS, per_label=1)
    for chain in (0, 1, 2):
        previous: set[str] = set()
        for n in (10, 25, 50, 100):
            _, labels = class_slice(splits, INTENTS, n, chain)
            assert len(labels) == n and previous <= set(labels)
            previous = set(labels)
    firsts = {tuple(class_slice(splits, INTENTS, 10, chain)[1]) for chain in (0, 1, 2)}
    assert len(firsts) == 3


def test_a_slice_keeps_every_example_of_its_intents_and_nothing_else():
    splits = toy_splits(INTENTS, per_label=3)
    sliced, labels = class_slice(splits, INTENTS, 25, 1)
    for name, df in sliced.items():
        assert set(df["label"]) == set(labels)
        assert len(df) == 3 * 25
        assert set(df["id"]) <= set(splits[name]["id"])


def test_subsets_are_stratified_and_nested():
    train = toy_splits(INTENTS[:5], per_label=40)["train"]
    previous: set[str] = set()
    for per_class in (1, 5, 10, 25):
        picked = subset(train, per_class, seed=3)
        assert picked["label"].value_counts().eq(per_class).all()
        assert previous <= set(picked["id"])
        previous = set(picked["id"])
    assert subset(train, None, seed=3) is train


def test_loading_detects_a_changed_split(data_dir):
    path = data_dir / "vast" / "test.jsonl"
    path.write_text(path.read_text().replace("for example 0.", "against example 0."))
    with pytest.raises(RuntimeError, match="checksum"):
        data.load("vast", "test", data_dir)


def test_few_shot_examples_cover_every_label_in_a_shuffled_order(data_dir):
    examples = few_shot_examples("vast", 5, seed=0, data_dir=data_dir)
    assert examples["label"].value_counts().to_dict() == {"against": 5, "for": 5, "neutral": 5}
    assert list(examples["label"]) != sorted(examples["label"])


def test_an_intent_has_the_same_few_shot_example_in_every_slice(data_dir):
    full = few_shot_examples(CLINC150, 1, seed=0, data_dir=data_dir)
    sliced = few_shot_examples(slice_name(10, 0), 1, seed=0, data_dir=data_dir)
    assert len(full) == 151 and OUT_OF_SCOPE in set(full["label"])
    assert sorted(sliced["label"]) == data.labels(slice_name(10, 0), data_dir)
    assert set(sliced["id"]) == set(full[full["label"].isin(sliced["label"])]["id"])
