"""calculator.py için risk-temelli testler.

İki bölüm: (1) make_cf fixture yardımcısı, (2) bilinen tuzakları yakalayan testler.
NaN karşılaştırması nan != nan olduğu için, NaN beklenen yerlerde np.isnan ya da
np.testing.assert_allclose(..., equal_nan=True) kullanılır.
"""

import numpy as np

from quality_engine.data.base import CompanyFinancials
from quality_engine.metrics.calculator import (
    gross_margin, operating_margin, fcf_margin, revenue_growth,
    roic, reinvestment_rate, rnd_to_revenue,
)


# --- BÖLÜM 1: make_cf fixture yardımcısı ---

def make_cf(**kwargs):
    """CompanyFinancials üretir: verilen alanları kullanır, verilmeyen 14 finansal
    alanı aynı uzunlukta [np.nan] listesiyle doldurur (zorunlu alanlar dolu kalsın)."""
    # verilen listelerden uzunluğu bul (ilk verilen list alanın uzunluğu)
    n = len(next(v for v in kwargs.values() if isinstance(v, list)))
    # 14 finansal alanın tam listesi:
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


# --- BÖLÜM 2: testler ---

# TEST 1 — capex işareti (BUG GEÇMİŞİ: capex toplanırsa FCF şişer)
def test_fcf_margin_capex_subtracted():
    # OCF=100, capex=10 (pozitif, provider normalize etti), revenue=200
    # Doğru: FCF=(100-10)=90, margin=90/200=0.45
    # Yanlış (toplama): (100+10)/200=0.55 — bu testi GEÇEMEZ
    cf = make_cf(operating_cash_flow=[100.0], capex=[10.0], revenue=[200.0])
    assert fcf_margin(cf) == [0.45]


# TEST 2 — ROIC pretax<=0 -> NaN
def test_roic_negative_pretax_is_nan():
    # pretax negatif -> tax rate tanımsız -> ROIC o dönem NaN
    cf = make_cf(ebit=[50.0], pretax_income=[-10.0], tax_provision=[2.0],
                 total_debt=[100.0], total_equity=[100.0], cash=[20.0])
    result = roic(cf)
    assert np.isnan(result[0])


# TEST 3 — ROIC tax clamp (oran 0.30'u aşmamalı)
def test_roic_tax_rate_clamped():
    # tax_provision/pretax = 50/100 = 0.50 -> clamp 0.30'a iner
    # NOPAT = ebit*(1-0.30) = 100*0.70 = 70
    # IC = 200+100-50 = 250 -> ROIC = 70/250 = 0.28
    cf = make_cf(ebit=[100.0], pretax_income=[100.0], tax_provision=[50.0],
                 total_debt=[200.0], total_equity=[100.0], cash=[50.0])
    np.testing.assert_allclose(roic(cf), [0.28], rtol=1e-9)


# TEST 4 — ROIC ic<=0 -> NaN
def test_roic_nonpositive_ic_is_nan():
    # cash > debt+equity -> IC negatif -> NaN
    cf = make_cf(ebit=[50.0], pretax_income=[100.0], tax_provision=[21.0],
                 total_debt=[10.0], total_equity=[10.0], cash=[100.0])
    assert np.isnan(roic(cf)[0])


# TEST 5 — reinvestment NOPAT<=0 -> NaN
def test_reinvestment_nonpositive_nopat_is_nan():
    # ebit negatif -> NOPAT<=0 -> reinvestment NaN
    cf = make_cf(ebit=[-50.0], pretax_income=[100.0], tax_provision=[21.0],
                 capex=[10.0], depreciation=[5.0], change_in_working_capital=[2.0])
    assert np.isnan(reinvestment_rate(cf)[0])


# TEST 6 — reinvestment NEGATİF sonuç GEÇERLİ (NaN OLMAMALI)
def test_reinvestment_negative_is_valid():
    # net_capex = capex-dep = 5-10 = -5; +ΔWC(-3) = -8; NOPAT pozitif
    # NOPAT = 100*(1-0.21)=79; reinvestment = -8/79 ≈ -0.101 (negatif, GEÇERLİ)
    cf = make_cf(ebit=[100.0], pretax_income=[100.0], tax_provision=[21.0],
                 capex=[5.0], depreciation=[10.0], change_in_working_capital=[-3.0])
    result = reinvestment_rate(cf)
    assert result[0] < 0  # negatif olmalı
    assert not np.isnan(result[0])  # NaN OLMAMALI


# TEST 7 — revenue_growth NaN lokal kalır (ileri doldurmaz)
def test_revenue_growth_nan_is_local():
    # revenue=[100, nan, 120]:
    # growth[0]=nan (tanımsız), growth[1]=nan (nan/100), growth[2]=nan (120/nan)
    # KRİTİK: growth[2] bir SAYI gelirse pandas NaN'i ileri doldurmuş demektir (YANLIŞ)
    cf = make_cf(revenue=[100.0, np.nan, 120.0])
    result = revenue_growth(cf)
    assert all(np.isnan(x) for x in result)


# TEST 8 — revenue_growth normal (akıl sağlığı)
def test_revenue_growth_normal():
    # revenue=[100,110,121]: growth=[nan, 0.10, 0.10]
    cf = make_cf(revenue=[100.0, 110.0, 121.0])
    np.testing.assert_allclose(revenue_growth(cf), [np.nan, 0.10, 0.10], equal_nan=True, rtol=1e-9)


# TEST 9 — sıfıra bölme koruması (gross margin, revenue=0 -> NaN)
def test_gross_margin_zero_revenue_is_nan():
    cf = make_cf(revenue=[0.0], cogs=[40.0])
    assert np.isnan(gross_margin(cf)[0])


# TEST 10 — rnd_to_revenue NaN propagate (R&D yok -> NaN)
def test_rnd_to_revenue_missing_is_nan():
    # rnd_expense NaN (banka gibi) -> sonuç NaN
    cf = make_cf(revenue=[100.0], rnd_expense=[np.nan])
    assert np.isnan(rnd_to_revenue(cf)[0])
