"""scoring.py için risk-temelli testler (yön düzeltmesi, grup kapsamı, yapı, varsayılan)."""

import numpy as np
import pandas as pd

from quality_engine.scoring import compute_qscore, _apply_directions, FACTOR_GROUPS


# TEST 1 — yön düzeltmesi: stability ve reinvestment 1-rank ile çevrilir,
# diğerleri DEĞİŞMEZ
def test_apply_directions():
    df = pd.DataFrame({
        "gross_margin_level_last": [0.8, 0.2],   # düz kalmalı
        "gross_margin_stability": [0.9, 0.1],    # 1-rank: 0.1, 0.9 olmalı
        "reinvestment_rate_level_last": [0.7, 0.3],  # 1-rank: 0.3, 0.7
        "roic_level_last": [0.6, 0.4],           # düz kalmalı
    }, index=["A", "B"])
    out = _apply_directions(df)
    # düz feature'lar değişmedi
    assert out["gross_margin_level_last"].tolist() == [0.8, 0.2]
    assert out["roic_level_last"].tolist() == [0.6, 0.4]
    # stability ters çevrildi (1-rank)
    np.testing.assert_allclose(out["gross_margin_stability"].tolist(), [0.1, 0.9])
    # reinvestment ters çevrildi
    np.testing.assert_allclose(out["reinvestment_rate_level_last"].tolist(), [0.3, 0.7])


# TEST 2 — faktör grupları tam 18 feature kapsıyor, çakışma yok
def test_factor_groups_cover_18():
    tum = [f for cols in FACTOR_GROUPS.values() for f in cols]
    assert len(tum) == 18              # toplam 18
    assert len(set(tum)) == 18         # çakışma yok (her feature tek grupta)


# TEST 3 — compute_qscore: çıktı yapısı, 0-100 aralığı, sıralama
def test_compute_qscore_structure():
    # 10 şirket, 18 feature, rastgele rank (0-1) — seed sabit
    rng = np.random.default_rng(42)
    cols = [f for cols in FACTOR_GROUPS.values() for f in cols]
    ranked = pd.DataFrame(rng.random((10, 18)),
                          index=[f"S{i}" for i in range(10)], columns=cols)
    result = compute_qscore(ranked, n_buckets=5)
    qscore = result["qscore"]
    # 0-100 aralığı
    assert qscore.min() == 0.0 and qscore.max() == 100.0
    # azalan sıralı (en yüksek başta)
    assert qscore.is_monotonic_decreasing
    # 4 faktör skoru
    assert result["factor_scores"].shape == (10, 4)
    # buckets var
    assert len(result["buckets"]) == 10


# TEST 4 — eşit ağırlık varsayılanı: weights None -> her faktör 0.25
def test_equal_weights_default():
    rng = np.random.default_rng(1)
    cols = [f for cols in FACTOR_GROUPS.values() for f in cols]
    ranked = pd.DataFrame(rng.random((20, 18)),
                          index=[f"S{i}" for i in range(20)], columns=cols)
    result = compute_qscore(ranked)
    # 4 faktör, her biri 0.25
    assert all(abs(w - 0.25) < 1e-9 for w in result["weights"].values())
    assert len(result["weights"]) == 4
