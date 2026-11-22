"""Evaluation metrics: Spearman correlation, precision@N, average precision, ablation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.stats import spearmanr


def spearman_correlation(x: Iterable[float], y: Iterable[float]) -> float:
    """Spearman rank correlation between two score lists."""
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if len(x_arr) < 2:
        return float("nan")
    corr, _ = spearmanr(x_arr, y_arr)
    return float(corr)


def rank_scores(scores: dict[str, float]) -> dict[str, int]:
    """Return rank (1 = best/highest score) for each item. Ties get average rank."""
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ranks: dict[str, float] = {}
    n = len(items)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][1] == items[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[items[k][0]] = avg_rank
        i = j + 1
    return {k: int(v) if v == int(v) else v for k, v in ranks.items()}


def precision_at_n(
    ranked_items: list[tuple[str, float]],
    relevant: set[str],
    n: int,
) -> float:
    """Precision@N: fraction of relevant items in top-N ranked by score."""
    if n == 0:
        return 0.0
    top_n = [item for item, _ in ranked_items[:n]]
    return sum(1 for item in top_n if item in relevant) / n


def average_precision(
    ranked_items: list[tuple[str, float]],
    relevant: set[str],
) -> float:
    """Average Precision (AveP) as used by Zhu et al. for ACL Fellows identification."""
    if not relevant:
        return 0.0
    precision_sum = 0.0
    relevant_found = 0
    for i, (item, _) in enumerate(ranked_items, start=1):
        if item in relevant:
            relevant_found += 1
            precision_sum += relevant_found / i
    return precision_sum / len(relevant)


def threshold_ablation(
    edges: list[tuple[str, str, float]],
    cited_paper_ids: Iterable[str],
    thresholds: list[float],
) -> dict[float, dict[str, float]]:
    """Compute weighted citation counts per cited paper at each weight threshold.

    Args:
        edges: list of (citing_paper_id, cited_paper_id, weight)
        cited_paper_ids: ids of papers that receive citations (the oeuvre)
        thresholds: list of weight cutoffs to evaluate

    Returns:
        dict mapping threshold -> {cited_paper_id: weighted_count}
    """
    cited_set = set(cited_paper_ids)
    results: dict[float, dict[str, float]] = {}
    for t in thresholds:
        wcp: dict[str, float] = {pid: 0.0 for pid in cited_set}
        for citing_id, cited_id, weight in edges:
            if weight >= t and cited_id in wcp:
                wcp[cited_id] += weight
        results[t] = wcp
    return results


def compare_indices(
    conventional: dict[str, int],
    weighted: dict[str, int],
    hip: dict[str, int],
) -> dict[str, float]:
    """Compare three h-index variants across the same set of authors.

    Returns Spearman correlations between each pair of methods.
    """
    authors = set(conventional) & set(weighted) & set(hip)
    if len(authors) < 2:
        return {"n_authors": float(len(authors))}

    c_scores = {a: float(conventional[a]) for a in authors}
    w_scores = {a: float(weighted[a]) for a in authors}
    h_scores = {a: float(hip[a]) for a in authors}

    return {
        "n_authors": float(len(authors)),
        "spearman_conventional_weighted": spearman_correlation(
            [c_scores[a] for a in authors], [w_scores[a] for a in authors]
        ),
        "spearman_conventional_hip": spearman_correlation(
            [c_scores[a] for a in authors], [h_scores[a] for a in authors]
        ),
        "spearman_weighted_hip": spearman_correlation(
            [w_scores[a] for a in authors], [h_scores[a] for a in authors]
        ),
    }
