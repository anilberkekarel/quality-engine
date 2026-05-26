"""Ham finansallardan quality metriklerini hesaplayan saf fonksiyonlar (Aşama 1: zaman serisi).

Her fonksiyon bir CompanyFinancials alır ve girdiyle aynı uzunlukta bir zaman
serisi (list[float]) döndürür. State yok, sınıf yok; sadece fonksiyonlar.
"""

import numpy as np
import pandas as pd

from ..data.base import CompanyFinancials

MIN_TAX_RATE = 0.0
MAX_TAX_RATE = 0.30


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


def _tax_rate(cf: CompanyFinancials) -> pd.Series:
    """Clamp'li efektif vergi oranı. pretax<=0 -> NaN.
    [MIN_TAX_RATE, MAX_TAX_RATE] aralığına clip."""
    # TODO (tax rate terfisi): Tek-dönem clamp'li efektif oranı → şirketin
    # CAUSAL geçmiş ortalama efektif oranına çevir (sadece o döneme kadarki
    # pozitif-pretax yıllardan; gelecekten ASLA — look-ahead bias).
    # Bu terfi iki sorunu birden çözer:
    #   (1) ana tax rate'i kararlı yapar (tek-dönem gürültüsünü giderir)
    #   (2) pretax≤0 ama EBIT>0 dönemlerini şirket-içi GERÇEK veriyle doldurur (imputation değil)
    # Yeterli causal geçmiş yoksa (ilk yıllar / kronik-zarar şirket) → NaN kalır (dürüst).
    pretax = pd.Series(cf.pretax_income, dtype="float64")
    tax_provision = pd.Series(cf.tax_provision, dtype="float64")
    # pretax <= 0 olan dönemleri NaN yap (efektif oran tanımsız):
    pretax_valid = pretax.where(pretax > 0, np.nan)
    tax_rate = tax_provision / pretax_valid
    # clamp (pandas .clip NaN'i KORUR, manuel min/max KULLANMA):
    return tax_rate.clip(MIN_TAX_RATE, MAX_TAX_RATE)


def _nopat(cf: CompanyFinancials) -> pd.Series:
    """NOPAT = EBIT * (1 - tax_rate). EBIT katı (fallback yok)."""
    ebit = pd.Series(cf.ebit, dtype="float64")
    return ebit * (1 - _tax_rate(cf))


def roic(cf: CompanyFinancials) -> list[float]:
    """ROIC = NOPAT / Invested Capital, her dönem için.

    NOPAT = EBIT * (1 - tax_rate); IC = total_debt + total_equity - cash
    (dönem-sonu değerleri, MVP). tax_rate = tax_provision / pretax_income,
    [MIN_TAX_RATE, MAX_TAX_RATE] aralığına clamp'lenir.

    NaN davranışı: pretax_income <= 0 ise efektif oran tanımsız → o dönem NaN;
    ic <= 0 ise (negatif/sıfır sermaye anlamsız) → NaN; herhangi bir bileşen
    (ebit, tax_rate, debt, equity, cash) NaN ise sonuç NaN (pandas propagate).
    """
    nopat = _nopat(cf)

    # PARÇA 3 — Invested Capital (dönem-sonu, MVP)
    # TODO (invested capital terfisi): dönem-sonu yerine ortalama IC kullan
    # (dönem başı + dönem sonu)/2 — bir dönem kaybı pahasına, teorik olarak doğru.
    total_debt = pd.Series(cf.total_debt, dtype="float64")
    total_equity = pd.Series(cf.total_equity, dtype="float64")
    cash = pd.Series(cf.cash, dtype="float64")
    ic = total_debt + total_equity - cash
    # ic <= 0 ise NaN (negatif/sıfır sermaye anlamsız):
    ic = ic.where(ic > 0, np.nan)

    # PARÇA 4 — Birleştir
    roic = nopat / ic
    roic = roic.replace([np.inf, -np.inf], np.nan)
    return roic.tolist()


def reinvestment_rate(cf: CompanyFinancials) -> list[float]:
    """Reinvestment rate = (net capex + ΔWC) / NOPAT, her dönem için (Damodaran).

    net_capex = capex - depreciation. capex provider tarafından POZİTİFE
    normalize edilmiş, depreciation (yfinance) POZİTİF; dolayısıyla net_capex
    NEGATİF olabilir — bu GEÇERLİ (asset-light şirket amortismandan az yatırım
    yapar). ΔWC kendi işaretiyle girer (abs alınmaz; işaret bilgi taşır).
    Sonuç da NEGATİF olabilir — bu da GEÇERLİ (sermaye serbest bırakma).

    NaN davranışı: NOPAT <= 0 ise payda anlamsız → o dönem NaN. Herhangi bir
    bileşen (capex, depreciation, ΔWC, NOPAT) NaN ise sonuç NaN (propagate).

    Not: Tek-yıl değeri ΔWC oynaklığı yüzünden gürültülüdür; asıl sinyal
    çok-yıllık trend/istikrardadır (Aşama 2).
    """
    capex = pd.Series(cf.capex, dtype="float64")
    depreciation = pd.Series(cf.depreciation, dtype="float64")
    change_in_working_capital = pd.Series(cf.change_in_working_capital, dtype="float64")

    nopat = _nopat(cf)

    net_capex = capex - depreciation
    numerator = net_capex + change_in_working_capital
    # NOPAT <= 0 olan dönemleri NaN yap (payda anlamsız):
    nopat_valid = nopat.where(nopat > 0, np.nan)

    reinvestment = numerator / nopat_valid
    reinvestment = reinvestment.replace([np.inf, -np.inf], np.nan)
    return reinvestment.tolist()


def rnd_to_revenue(cf: CompanyFinancials) -> list[float]:
    """Ar-Ge yoğunluğu: rnd_expense / revenue, her dönem için.

    R&D raporlamayan sektörlerde (banka, enerji) rnd_expense NaN gelir ve
    sonuç NaN döner — bu DOĞRU davranış (imputation yok); sektör-koşullu
    kullanım Aşama 2'de ele alınır. revenue NaN ise o dönem NaN; revenue == 0
    ise NaN.
    """
    revenue = pd.Series(cf.revenue, dtype="float64")
    rnd_expense = pd.Series(cf.rnd_expense, dtype="float64")
    revenue_nonzero = revenue.replace(0, np.nan)
    ratio = rnd_expense / revenue_nonzero
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    return ratio.tolist()
