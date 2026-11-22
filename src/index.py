"""Citation impact indices: conventional h-index, hip-index, weighted h-index."""

from __future__ import annotations

from collections.abc import Iterable


def h_index(citation_counts: Iterable[int]) -> int:
    """Conventional h-index: largest h with at least h papers cited >= h times."""
    counts = sorted((int(c) for c in citation_counts), reverse=True)
    h = 0
    for i, c in enumerate(counts, start=1):
        if c >= i:
            h = i
        else:
            break
    return h


def hip_index(
    citation_counts: Iterable[int],
    mention_counts: Iterable[int],
) -> int:
    """Hip-index (Zhu et al. 2015): citation edge weighted by c^2 (mention count squared).

    Args:
        citation_counts: number of citing papers per paper (unused in weighting, kept for API symmetry).
        mention_counts: in-body mention count c per citation edge; weight = c^2.

    Returns:
        Largest h such that at least h papers have weighted citation count >= h.
    """
    weights = sorted((int(c) ** 2 for c in mention_counts), reverse=True)
    h = 0
    for i, w in enumerate(weights, start=1):
        if w >= i:
            h = i
        else:
            break
    return h


def weighted_h_index(weighted_counts: Iterable[float]) -> int:
    """Weighted h-index: largest h with at least h papers having weighted citation count >= h."""
    counts = sorted((float(c) for c in weighted_counts), reverse=True)
    h = 0
    for i, c in enumerate(counts, start=1):
        if c >= i:
            h = i
        else:
            break
    return h
