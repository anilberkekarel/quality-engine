"""features.py (seri_ozetle) için risk-temelli testler.

Aşama 2 özetleme: bilinen tuzakları yakalayan minimum test seti.
"""

import numpy as np

from quality_engine.data.base import CompanyFinancials
from quality_engine.metrics.features import seri_ozetle, extract_features


def make_cf(**kwargs):
    """CompanyFinancials üretir: verilen alanları kullanır, verilmeyen 14 finansal
    alanı aynı uzunlukta [np.nan] listesiyle doldurur (zorunlu alanlar dolu kalsın)."""
    n = len(next(v for v in kwargs.values() if isinstance(v, list)))
    fields = ["revenue", "cogs", "operating_income", "ebit", "pretax_income",
              "tax_provision", "rnd_expense", "depreciation", "total_debt",
              "total_equity", "cash", "operating_cash_flow", "capex",
              "change_in_working_capital"]
    data = {f: kwargs.get(f, [np.nan] * n) for f in fields}
    return CompanyFinancials(
        ticker=kwargs.get("ticker", "TEST"),
        period_end_dates=list(range(n)),
        **data,
    )


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


# extract_features yapı testi: iki bölme, doğru anahtar sayıları, ayrım
def test_extract_features_structure():
    # basit dolu cf (5 dönem), tüm metrikler hesaplanabilsin
    cf = make_cf(
        revenue=[100.0,110.0,120.0,130.0,140.0],
        cogs=[40.0,44.0,48.0,52.0,56.0],
        operating_income=[30.0,33.0,36.0,39.0,42.0],
        ebit=[30.0,33.0,36.0,39.0,42.0],
        pretax_income=[28.0,31.0,34.0,37.0,40.0],
        tax_provision=[6.0,7.0,7.0,8.0,8.0],
        rnd_expense=[8.0,9.0,10.0,11.0,12.0],
        depreciation=[5.0,5.0,5.0,5.0,5.0],
        total_debt=[50.0,50.0,50.0,50.0,50.0],
        total_equity=[100.0,110.0,120.0,130.0,140.0],
        cash=[20.0,20.0,20.0,20.0,20.0],
        operating_cash_flow=[35.0,38.0,41.0,44.0,47.0],
        capex=[10.0,10.0,10.0,10.0,10.0],
        change_in_working_capital=[2.0,2.0,2.0,2.0,2.0],
    )
    result = extract_features(cf)
    # iki bölme var mı
    assert "features" in result and "meta" in result
    # 7 metrik × 3 feature = 21
    assert len(result["features"]) == 21
    # 7 metrik × 2 meta = 14
    assert len(result["meta"]) == 14
    # KRİTİK: meta anahtarları features'a sızmamış (leakage kontrolü)
    assert not any("_r2" in k or "_n_valid" in k for k in result["features"])
    # feature anahtarları meta'ya sızmamış
    assert all("_r2" in k or "_n_valid" in k for k in result["meta"])
