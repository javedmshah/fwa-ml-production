"""Fast unit tests that need no real data (safe for CI)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fwa.evaluate import precision_at_k, rank_providers  # noqa: E402
from fwa.split import assert_provider_disjoint, make_provider_split  # noqa: E402


@pytest.fixture
def labels():
    rng = np.random.default_rng(1)
    n = 1000
    return pd.DataFrame({"Provider": [f"PRV{i:05d}" for i in range(n)],
                         "PotentialFraud": np.where(rng.random(n) < 0.1, "Yes", "No")})


def test_split_is_provider_disjoint_and_complete(labels):
    split = make_provider_split(labels, seed=42)
    assert len(split) == len(labels)
    assert split["Provider"].is_unique
    assert_provider_disjoint(split)


def test_split_is_reproducible(labels):
    a = make_provider_split(labels, seed=42)
    b = make_provider_split(labels, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_split_keeps_fraud_rate_similar(labels):
    split = make_provider_split(labels, seed=42)
    rates = split.groupby("split")["label"].mean()
    assert rates.max() - rates.min() < 0.03


def test_duplicate_providers_rejected(labels):
    with pytest.raises(ValueError):
        make_provider_split(pd.concat([labels, labels.head(1)]), seed=42)


def test_overlap_detected():
    bad = pd.DataFrame({"Provider": ["A", "A"], "label": [0, 0], "split": ["train", "test"]})
    with pytest.raises(AssertionError):
        assert_provider_disjoint(bad)


def test_ranking_sorted_and_ties_deterministic():
    ids = pd.Series(["P3", "P1", "P2", "P4"])
    ranked = rank_providers(ids, np.array([0.5, 0.9, 0.5, 0.1]), limit=3)
    assert ranked["Provider"].tolist() == ["P1", "P2", "P3"]  # P2 before P3 on the tie
    assert ranked["rank"].tolist() == [1, 2, 3]


def test_precision_at_k():
    ids = pd.Series(["A", "B", "C", "D"])
    assert precision_at_k(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1]), ids, k=2) == 0.5
