"""Tests for citation impact indices."""


from src.index import h_index, hip_index, weighted_h_index


def test_h_index_basic():
    assert h_index([10, 8, 5, 4, 3]) == 4
    assert h_index([1, 1, 1]) == 1
    assert h_index([0, 0, 0]) == 0
    assert h_index([]) == 0
    assert h_index([100]) == 1


def test_weighted_h_index():
    # All weights equal to citation counts -> same as conventional
    assert weighted_h_index([10.0, 8.0, 5.0, 4.0, 3.0]) == 4
    # Fractional weights
    assert weighted_h_index([4.5, 4.5, 4.5, 4.5]) == 4
    assert weighted_h_index([0.5, 0.5, 0.5]) == 0


def test_hip_index():
    # c^2 weighting
    mention_counts = [1, 2, 3]
    # weights: 1, 4, 9 -> h-index of [9,4,1] = 3 (9>=3, 4>=2, 1>=1 fails? actually 1<3 so h=2)
    # sorted desc: 9,4,1 -> h: 9>=1 yes h=1, 4>=2 yes h=2, 1>=3 no -> h=2
    assert hip_index([0, 0, 0], mention_counts) == 2
