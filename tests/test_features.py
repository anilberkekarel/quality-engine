"""Risk-based tests for features.py (summarize_series).

Stage 2 summarization: a minimum test set that catches the known traps.
"""

import numpy as np

from quality_engine.data.base import CompanyFinancials
from quality_engine.metrics.features import summarize_series, extract_features


def make_cf(**kwargs):
    """Build a CompanyFinancials: use the given fields and fill the other 14
    financial fields with a same-length [np.nan] list (so required fields
    stay populated)."""
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


# TEST 1 — known slope (regular increase -> exact slope, R²=1)
def test_summarize_series_known_slope():
    # [0.10,0.20,0.30] regular 0.10 increase -> trend=0.10, r2=1.0, n_valid=3
    r = summarize_series([0.10, 0.20, 0.30])
    np.testing.assert_allclose(r["trend"], 0.10, rtol=1e-9)
    np.testing.assert_allclose(r["trend_r2"], 1.0, rtol=1e-9)
    assert r["level_last"] == 0.30
    assert r["n_valid"] == 3


# TEST 2 — POSITION PRESERVATION (NaN is skipped but x position is preserved)
# CRITICAL: if range is used instead of np.where, this test breaks
def test_summarize_series_position_preserved():
    # [nan,0.10,0.20,0.30]: non-NaN x=[1,2,3], y=[0.10,0.20,0.30]
    # slope is still 0.10 (because position is preserved), n_valid=3
    r = summarize_series([np.nan, 0.10, 0.20, 0.30])
    np.testing.assert_allclose(r["trend"], 0.10, rtol=1e-9)
    assert r["n_valid"] == 3
    assert r["level_last"] == 0.30


# TEST 3 — threshold: exactly 2 points (stability EXISTS, trend NaN)
def test_summarize_series_two_points_no_trend():
    r = summarize_series([0.10, 0.20])
    assert not np.isnan(r["stability"])  # >=2 -> stability exists
    assert np.isnan(r["trend"])          # <3 -> trend NaN
    assert r["n_valid"] == 2


# TEST 4 — threshold: exactly 1 point (only level_last)
def test_summarize_series_one_point_only_level():
    r = summarize_series([0.5])
    assert r["level_last"] == 0.5
    assert np.isnan(r["stability"])
    assert np.isnan(r["trend"])
    assert r["n_valid"] == 1


# TEST 5 — all NaN (edge: must not crash, everything NaN, n_valid=0)
def test_summarize_series_all_nan():
    r = summarize_series([np.nan, np.nan, np.nan])
    assert np.isnan(r["level_last"])
    assert np.isnan(r["trend"])
    assert np.isnan(r["stability"])
    assert r["n_valid"] == 0


# extract_features structure test: two compartments, expected key counts, separation
def test_extract_features_structure():
    # simple populated cf (5 periods), every metric computable
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
    # both compartments present
    assert "features" in result and "meta" in result
    # 7 metrics × 3 features = 21
    assert len(result["features"]) == 21
    # 7 metrics × 2 meta = 14
    assert len(result["meta"]) == 14
    # CRITICAL: meta keys did not leak into features (leakage check)
    assert not any("_r2" in k or "_n_valid" in k for k in result["features"])
    # feature keys did not leak into meta
    assert all("_r2" in k or "_n_valid" in k for k in result["meta"])
