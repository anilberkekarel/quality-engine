"""Risk-based tests for calculator.py.

Two sections: (1) make_cf fixture helper, (2) tests that catch known traps.
Since nan != nan, np.isnan or np.testing.assert_allclose(..., equal_nan=True)
is used wherever a NaN is expected.
"""

import numpy as np

from quality_engine.data.base import CompanyFinancials
from quality_engine.metrics.calculator import (
    gross_margin, operating_margin, fcf_margin, revenue_growth,
    roic, reinvestment_rate, rnd_to_revenue,
)


# --- SECTION 1: make_cf fixture helper ---

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


# --- SECTION 2: tests ---

# TEST 1 — capex sign (BUG HISTORY: if capex is added, FCF gets inflated)
def test_fcf_margin_capex_subtracted():
    # OCF=100, capex=10 (positive, normalized by provider), revenue=200
    # Correct: FCF=(100-10)=90, margin=90/200=0.45
    # Wrong (adding): (100+10)/200=0.55 — this test would NOT pass
    cf = make_cf(operating_cash_flow=[100.0], capex=[10.0], revenue=[200.0])
    assert fcf_margin(cf) == [0.45]


# TEST 2 — ROIC pretax<=0 -> NaN
def test_roic_negative_pretax_is_nan():
    # pretax negative -> tax rate undefined -> ROIC NaN for that period
    cf = make_cf(ebit=[50.0], pretax_income=[-10.0], tax_provision=[2.0],
                 total_debt=[100.0], total_equity=[100.0], cash=[20.0])
    result = roic(cf)
    assert np.isnan(result[0])


# TEST 3 — ROIC tax clamp (rate must not exceed 0.30)
def test_roic_tax_rate_clamped():
    # tax_provision/pretax = 50/100 = 0.50 -> clamped down to 0.30
    # NOPAT = ebit*(1-0.30) = 100*0.70 = 70
    # IC = 200+100-50 = 250 -> ROIC = 70/250 = 0.28
    cf = make_cf(ebit=[100.0], pretax_income=[100.0], tax_provision=[50.0],
                 total_debt=[200.0], total_equity=[100.0], cash=[50.0])
    np.testing.assert_allclose(roic(cf), [0.28], rtol=1e-9)


# TEST 4 — ROIC ic<=0 -> NaN
def test_roic_nonpositive_ic_is_nan():
    # cash > debt+equity -> IC negative -> NaN
    cf = make_cf(ebit=[50.0], pretax_income=[100.0], tax_provision=[21.0],
                 total_debt=[10.0], total_equity=[10.0], cash=[100.0])
    assert np.isnan(roic(cf)[0])


# TEST 5 — reinvestment NOPAT<=0 -> NaN
def test_reinvestment_nonpositive_nopat_is_nan():
    # ebit negative -> NOPAT<=0 -> reinvestment NaN
    cf = make_cf(ebit=[-50.0], pretax_income=[100.0], tax_provision=[21.0],
                 capex=[10.0], depreciation=[5.0], change_in_working_capital=[2.0])
    assert np.isnan(reinvestment_rate(cf)[0])


# TEST 6 — NEGATIVE reinvestment result is VALID (MUST NOT be NaN)
def test_reinvestment_negative_is_valid():
    # net_capex = capex-dep = 5-10 = -5; +ΔWC(-3) = -8; NOPAT positive
    # NOPAT = 100*(1-0.21)=79; reinvestment = -8/79 ≈ -0.101 (negative, VALID)
    cf = make_cf(ebit=[100.0], pretax_income=[100.0], tax_provision=[21.0],
                 capex=[5.0], depreciation=[10.0], change_in_working_capital=[-3.0])
    result = reinvestment_rate(cf)
    assert result[0] < 0  # must be negative
    assert not np.isnan(result[0])  # MUST NOT be NaN


# TEST 7 — revenue_growth NaN stays local (no forward-fill)
def test_revenue_growth_nan_is_local():
    # revenue=[100, nan, 120]:
    # growth[0]=nan (undefined), growth[1]=nan (nan/100), growth[2]=nan (120/nan)
    # CRITICAL: if growth[2] becomes a number, pandas forward-filled NaN (WRONG)
    cf = make_cf(revenue=[100.0, np.nan, 120.0])
    result = revenue_growth(cf)
    assert all(np.isnan(x) for x in result)


# TEST 8 — revenue_growth normal (sanity)
def test_revenue_growth_normal():
    # revenue=[100,110,121]: growth=[nan, 0.10, 0.10]
    cf = make_cf(revenue=[100.0, 110.0, 121.0])
    np.testing.assert_allclose(revenue_growth(cf), [np.nan, 0.10, 0.10], equal_nan=True, rtol=1e-9)


# TEST 9 — divide-by-zero guard (gross margin, revenue=0 -> NaN)
def test_gross_margin_zero_revenue_is_nan():
    cf = make_cf(revenue=[0.0], cogs=[40.0])
    assert np.isnan(gross_margin(cf)[0])


# TEST 10 — rnd_to_revenue NaN propagates (no R&D -> NaN)
def test_rnd_to_revenue_missing_is_nan():
    # rnd_expense NaN (e.g. a bank) -> result NaN
    cf = make_cf(revenue=[100.0], rnd_expense=[np.nan])
    assert np.isnan(rnd_to_revenue(cf)[0])
