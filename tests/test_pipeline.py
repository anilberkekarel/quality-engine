"""pipeline.build_feature_matrix için RESILIENCE testleri.

Gerçek yfinance'a gitmeden, kontrollü sahte (mock) provider ile davranışı
doğrular: hatalı/boş şirketler elenir, geçerliler matriste kalır, ve
feature/meta ayrımı (leakage önlemi) korunur.
"""

import numpy as np

from quality_engine.data.base import CompanyFinancials, DataProvider
from quality_engine.pipeline import build_feature_matrix


class MockProvider(DataProvider):
    """Test için sahte provider. Önceden tanımlı senaryolar döndürür:
    - 'GOOD': dolu, geçerli CompanyFinancials
    - 'EMPTY': boş CompanyFinancials (geçersiz ticker simülasyonu)
    - 'ERROR': exception fırlatır (yfinance hatası simülasyonu)
    """
    def get_financials(self, ticker: str) -> CompanyFinancials:
        if ticker == "ERROR":
            raise ValueError("simulated fetch error")
        if ticker == "EMPTY":
            return CompanyFinancials(
                ticker=ticker, period_end_dates=[],
                revenue=[], cogs=[], operating_income=[], ebit=[],
                pretax_income=[], tax_provision=[], rnd_expense=[],
                depreciation=[], total_debt=[], total_equity=[], cash=[],
                operating_cash_flow=[], capex=[], change_in_working_capital=[],
            )
        return CompanyFinancials(
            ticker=ticker, period_end_dates=[0,1,2,3,4],
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


# TEST 1 — geçersiz/boş/hatalı şirketler elenir, geçerliler kalır
def test_pipeline_resilience():
    tickers = ["GOOD1", "EMPTY", "GOOD2", "ERROR", "GOOD3"]
    fdf, mdf = build_feature_matrix(tickers, MockProvider(), delay=0.0)
    # 5 ticker'dan 3 GOOD kalmalı (EMPTY ve ERROR elendi)
    assert list(fdf.index) == ["GOOD1", "GOOD2", "GOOD3"]
    assert fdf.shape == (3, 21)   # 3 şirket, 21 feature
    assert mdf.shape == (3, 14)   # 3 şirket, 14 meta
    # EMPTY ve ERROR matriste YOK (resilience)
    assert "EMPTY" not in fdf.index
    assert "ERROR" not in fdf.index


# TEST 2 — leakage ayrımı korunuyor (feature/meta sütunları karışmamış)
def test_pipeline_no_leakage():
    fdf, mdf = build_feature_matrix(["GOOD1"], MockProvider(), delay=0.0)
    # feature matrisinde meta sütunu (r2, n_valid) OLMAMALI
    assert not any("_r2" in c or "_n_valid" in c for c in fdf.columns)
    # meta matrisinde sadece meta sütunları olmalı
    assert all("_r2" in c or "_n_valid" in c for c in mdf.columns)
