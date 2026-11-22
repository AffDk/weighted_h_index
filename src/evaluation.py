"""Evaluation metrics: correlation, precision@N, ablation."""

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


def precision_at_n(
    ranked_scores: list[tuple[str, float]],
    relevant: set[str],
    n: int,
) -> float:
    """Precision@N: fraction of relevant items in top-N ranked by score."""
    top_n = [item for item, _ in ranked_scores[:n]]
    if n == 0:
        return 0.0
    return sum(1 for item in top_n if item in relevant) / n
