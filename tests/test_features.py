"""features.py (seri_ozetle) için risk-temelli testler.

Aşama 2 özetleme: bilinen tuzakları yakalayan minimum test seti.
"""

import numpy as np

from quality_engine.metrics.features import seri_ozetle


# TEST 1 — bilinen eğim (düzenli artış -> tam eğim, R²=1)
def test_seri_ozetle_known_slope():
    # [0.10,0.20,0.30] düzenli 0.10 artış -> trend=0.10, r2=1.0, n_valid=3
    r = seri_ozetle([0.10, 0.20, 0.30])
    np.testing.assert_allclose(r["trend"], 0.10, rtol=1e-9)
    np.testing.assert_allclose(r["trend_r2"], 1.0, rtol=1e-9)
    assert r["level_last"] == 0.30
    assert r["n_valid"] == 3


# TEST 2 — POZİSYON KORUMASI (NaN atlanır ama x pozisyonu korunur)
# KRİTİK: np.where yerine range kullanılırsa bu test patlar
def test_seri_ozetle_position_preserved():
    # [nan,0.10,0.20,0.30]: dolu x=[1,2,3], y=[0.10,0.20,0.30]
    # eğim hâlâ 0.10 (pozisyon korunduğu için), n_valid=3
    r = seri_ozetle([np.nan, 0.10, 0.20, 0.30])
    np.testing.assert_allclose(r["trend"], 0.10, rtol=1e-9)
    assert r["n_valid"] == 3
    assert r["level_last"] == 0.30


# TEST 3 — eşik: tam 2 nokta (stability VAR, trend NaN)
def test_seri_ozetle_two_points_no_trend():
    r = seri_ozetle([0.10, 0.20])
    assert not np.isnan(r["stability"])  # >=2 -> stability var
    assert np.isnan(r["trend"])          # <3 -> trend NaN
    assert r["n_valid"] == 2


# TEST 4 — eşik: tam 1 nokta (sadece level_last)
def test_seri_ozetle_one_point_only_level():
    r = seri_ozetle([0.5])
    assert r["level_last"] == 0.5
    assert np.isnan(r["stability"])
    assert np.isnan(r["trend"])
    assert r["n_valid"] == 1


# TEST 5 — hepsi NaN (kenar: patlamamalı, her şey NaN, n_valid=0)
def test_seri_ozetle_all_nan():
    r = seri_ozetle([np.nan, np.nan, np.nan])
    assert np.isnan(r["level_last"])
    assert np.isnan(r["trend"])
    assert np.isnan(r["stability"])
    assert r["n_valid"] == 0
