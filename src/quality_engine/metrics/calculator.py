"""Ham finansallardan quality metriklerini hesaplayan saf fonksiyonlar (Aşama 1: zaman serisi).

Her fonksiyon bir CompanyFinancials alır ve girdiyle aynı uzunlukta bir zaman
serisi (list[float]) döndürür. State yok, sınıf yok; sadece fonksiyonlar.
"""

import numpy as np
import pandas as pd

from ..data.base import CompanyFinancials


def gross_margin(cf: CompanyFinancials) -> list[float]:
    """Brüt marj: (revenue - cogs) / revenue, her dönem için.

    revenue veya cogs NaN ise o dönem NaN; revenue == 0 ise o dönem NaN.
    """
    revenue = pd.Series(cf.revenue, dtype="float64")
    cogs = pd.Series(cf.cogs, dtype="float64")
    revenue_nonzero = revenue.replace(0, np.nan)
    margin = (revenue - cogs) / revenue_nonzero
    margin = margin.replace([np.inf, -np.inf], np.nan)
    return margin.tolist()


def operating_margin(cf: CompanyFinancials) -> list[float]:
    """Faaliyet marjı: operating_income / revenue, her dönem için.

    operating_income veya revenue NaN ise o dönem NaN; revenue == 0 ise NaN.
    """
    revenue = pd.Series(cf.revenue, dtype="float64")
    operating_income = pd.Series(cf.operating_income, dtype="float64")
    revenue_nonzero = revenue.replace(0, np.nan)
    margin = operating_income / revenue_nonzero
    margin = margin.replace([np.inf, -np.inf], np.nan)
    return margin.tolist()


def fcf_margin(cf: CompanyFinancials) -> list[float]:
    """Serbest nakit akışı marjı: (operating_cash_flow - capex) / revenue.

    capex provider tarafından pozitife normalize edildiği için ÇIKARILIR.
    Herhangi bir bileşen (operating_cash_flow, capex, revenue) NaN ise o dönem
    NaN; revenue == 0 ise NaN.
    """
    revenue = pd.Series(cf.revenue, dtype="float64")
    operating_cash_flow = pd.Series(cf.operating_cash_flow, dtype="float64")
    capex = pd.Series(cf.capex, dtype="float64")
    revenue_nonzero = revenue.replace(0, np.nan)
    fcf = operating_cash_flow - capex
    margin = fcf / revenue_nonzero
    margin = margin.replace([np.inf, -np.inf], np.nan)
    return margin.tolist()


def revenue_growth(cf: CompanyFinancials) -> list[float]:
    """Gelir büyümesi: (revenue[t] - revenue[t-1]) / revenue[t-1].

    İlk eleman her zaman NaN (büyüme tanımsız). revenue[t] veya revenue[t-1]
    NaN ise o dönem NaN; revenue[t-1] == 0 ise NaN. NaN lokal kalır.
    """
    revenue = pd.Series(cf.revenue, dtype="float64")
    # fill_method=None: NaN'leri forward-fill ETME; aksi halde bir NaN'in
    # önceki değeri sonraki döneme sızar ve "NaN lokal kalsın" kuralı bozulur.
    growth = revenue.pct_change(fill_method=None)
    growth = growth.replace([np.inf, -np.inf], np.nan)
    return growth.tolist()
