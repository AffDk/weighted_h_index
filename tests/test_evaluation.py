"""Tests for evaluation metrics."""

from src.evaluation import (
    average_precision,
    compare_indices,
    precision_at_n,
    rank_scores,
    spearman_correlation,
    threshold_ablation,
)


def test_spearman_correlation():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert abs(spearman_correlation(x, y) - 1.0) < 1e-10

    y_rev = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert abs(spearman_correlation(x, y_rev) - (-1.0)) < 1e-10


def test_spearman_single():
    import math

    assert math.isnan(spearman_correlation([1.0], [1.0]))


def test_rank_scores():
    scores = {"a": 10.0, "b": 5.0, "c": 5.0, "d": 1.0}
    ranks = rank_scores(scores)
    assert ranks["a"] == 1
    assert ranks["b"] == 2.5  # tied with c
    assert ranks["c"] == 2.5
    assert ranks["d"] == 4


def test_precision_at_n():
    ranked = [("a", 1.0), ("b", 0.9), ("c", 0.8), ("d", 0.7)]
    relevant = {"a", "c"}
    assert precision_at_n(ranked, relevant, 2) == 0.5  # a in top 2
    assert precision_at_n(ranked, relevant, 4) == 0.5  # a, c in top 4
    assert precision_at_n(ranked, relevant, 0) == 0.0


def test_average_precision():
    ranked = [("a", 1.0), ("b", 0.9), ("c", 0.8), ("d", 0.7)]
    relevant = {"a", "c"}
    # a at rank 1: precision@1 = 1/1 = 1.0
    # c at rank 3: precision@3 = 2/3 = 0.667
    # AveP = (1.0 + 0.667) / 2 = 0.833
    ap = average_precision(ranked, relevant)
    assert abs(ap - 0.8333) < 0.01


def test_compare_indices():
    conventional = {"a": 3, "b": 2, "c": 1}
    weighted = {"a": 3, "b": 2, "c": 1}
    hip = {"a": 3, "b": 2, "c": 1}
    result = compare_indices(conventional, weighted, hip)
    assert result["spearman_conventional_weighted"] == 1.0
    assert result["n_authors"] == 3.0


def test_threshold_ablation():
    edges = [
        ("c1", "p1", 0.9),
        ("c2", "p1", 0.4),
        ("c3", "p2", 0.7),
    ]
    result = threshold_ablation(edges, ["p1", "p2"], thresholds=[0.0, 0.5, 0.8])
    assert result[0.0]["p1"] == 1.3  # 0.9 + 0.4
    assert result[0.5]["p1"] == 0.9  # only 0.9 >= 0.5
    assert result[0.8]["p1"] == 0.9  # only 0.9 >= 0.8
    assert result[0.5]["p2"] == 0.7  # 0.7 >= 0.5
    assert result[0.8]["p2"] == 0.0  # 0.7 < 0.8
