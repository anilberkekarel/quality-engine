"""Risk-based tests for scoring.py (direction correction, group coverage, structure, default)."""

import numpy as np
import pandas as pd

from quality_engine.scoring import compute_qscore, _apply_directions, FACTOR_GROUPS


# TEST 1 — direction correction: stability and reinvestment get inverted (1-rank),
# the others STAY UNCHANGED
def test_apply_directions():
    df = pd.DataFrame({
        "gross_margin_level_last": [0.8, 0.2],   # should stay unchanged
        "gross_margin_stability": [0.9, 0.1],    # 1-rank: should become 0.1, 0.9
        "reinvestment_rate_level_last": [0.7, 0.3],  # 1-rank: 0.3, 0.7
        "roic_level_last": [0.6, 0.4],           # should stay unchanged
    }, index=["A", "B"])
    out = _apply_directions(df)
    # unchanged features did not change
    assert out["gross_margin_level_last"].tolist() == [0.8, 0.2]
    assert out["roic_level_last"].tolist() == [0.6, 0.4]
    # stability got inverted (1-rank)
    np.testing.assert_allclose(out["gross_margin_stability"].tolist(), [0.1, 0.9])
    # reinvestment got inverted
    np.testing.assert_allclose(out["reinvestment_rate_level_last"].tolist(), [0.3, 0.7])


# TEST 2 — factor groups cover exactly 18 features, no overlap
def test_factor_groups_cover_18():
    all_features = [f for cols in FACTOR_GROUPS.values() for f in cols]
    assert len(all_features) == 18              # 18 total
    assert len(set(all_features)) == 18         # no overlap (every feature in exactly one group)


# TEST 3 — compute_qscore: output structure, 0-100 range, sort order
def test_compute_qscore_structure():
    # 10 companies, 18 features, random rank (0-1) — fixed seed
    rng = np.random.default_rng(42)
    cols = [f for cols in FACTOR_GROUPS.values() for f in cols]
    ranked = pd.DataFrame(rng.random((10, 18)),
                          index=[f"S{i}" for i in range(10)], columns=cols)
    result = compute_qscore(ranked, n_buckets=5)
    qscore = result["qscore"]
    # 0-100 range
    assert qscore.min() == 0.0 and qscore.max() == 100.0
    # sorted descending (highest first)
    assert qscore.is_monotonic_decreasing
    # 4 factor scores
    assert result["factor_scores"].shape == (10, 4)
    # buckets present
    assert len(result["buckets"]) == 10


# TEST 4 — equal-weight default: weights None -> 0.25 per factor
def test_equal_weights_default():
    rng = np.random.default_rng(1)
    cols = [f for cols in FACTOR_GROUPS.values() for f in cols]
    ranked = pd.DataFrame(rng.random((20, 18)),
                          index=[f"S{i}" for i in range(20)], columns=cols)
    result = compute_qscore(ranked)
    # 4 factors, 0.25 each
    assert all(abs(w - 0.25) < 1e-9 for w in result["weights"].values())
    assert len(result["weights"]) == 4
