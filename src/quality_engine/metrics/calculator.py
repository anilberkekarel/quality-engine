"""Pure functions that compute quality metrics from raw financials (Stage 1: time series).

Each function takes a CompanyFinancials and returns a time series
(list[float]) of the same length as the input. No state, no classes — only
functions.
"""

import numpy as np
import pandas as pd

from ..data.base import CompanyFinancials

MIN_TAX_RATE = 0.0
MAX_TAX_RATE = 0.30


def gross_margin(cf: CompanyFinancials) -> list[float]:
    """Gross margin: (revenue - cogs) / revenue, per period.

    If revenue or cogs is NaN the period is NaN; if revenue == 0 the period is NaN.
    """
    revenue = pd.Series(cf.revenue, dtype="float64")
    cogs = pd.Series(cf.cogs, dtype="float64")
    revenue_nonzero = revenue.replace(0, np.nan)
    margin = (revenue - cogs) / revenue_nonzero
    margin = margin.replace([np.inf, -np.inf], np.nan)
    return margin.tolist()


def operating_margin(cf: CompanyFinancials) -> list[float]:
    """Operating margin: operating_income / revenue, per period.

    If operating_income or revenue is NaN the period is NaN; if revenue == 0 it is NaN.
    """
    revenue = pd.Series(cf.revenue, dtype="float64")
    operating_income = pd.Series(cf.operating_income, dtype="float64")
    revenue_nonzero = revenue.replace(0, np.nan)
    margin = operating_income / revenue_nonzero
    margin = margin.replace([np.inf, -np.inf], np.nan)
    return margin.tolist()


def fcf_margin(cf: CompanyFinancials) -> list[float]:
    """Free cash flow margin: (operating_cash_flow - capex) / revenue.

    capex is normalized to positive by the provider, so it is SUBTRACTED.
    If any component (operating_cash_flow, capex, revenue) is NaN the
    period is NaN; if revenue == 0 it is NaN.
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
    """Revenue growth: (revenue[t] - revenue[t-1]) / revenue[t-1].

    The first element is always NaN (growth undefined). If revenue[t] or
    revenue[t-1] is NaN the period is NaN; if revenue[t-1] == 0 it is NaN.
    NaN stays local.
    """
    revenue = pd.Series(cf.revenue, dtype="float64")
    # fill_method=None: do NOT forward-fill NaNs; otherwise a NaN's previous
    # value would leak into the next period and break the "NaN stays local" rule.
    growth = revenue.pct_change(fill_method=None)
    growth = growth.replace([np.inf, -np.inf], np.nan)
    return growth.tolist()


def _tax_rate(cf: CompanyFinancials) -> pd.Series:
    """Clamped effective tax rate. pretax<=0 -> NaN.
    Clipped to the [MIN_TAX_RATE, MAX_TAX_RATE] range."""
    # TODO (tax rate upgrade): turn the single-period clamped effective rate
    # into the company's CAUSAL trailing-mean effective rate (only over
    # positive-pretax years up to that period; NEVER use the future —
    # look-ahead bias).
    # This upgrade solves two problems at once:
    #   (1) it stabilizes the main tax rate (removes single-period noise)
    #   (2) it fills pretax<=0 but EBIT>0 periods using the company's OWN
    #       historical data (not imputation)
    # If there is not enough causal history (early years / chronically
    # loss-making company) -> stays NaN (honest).
    pretax = pd.Series(cf.pretax_income, dtype="float64")
    tax_provision = pd.Series(cf.tax_provision, dtype="float64")
    # Periods where pretax <= 0 become NaN (effective rate undefined):
    pretax_valid = pretax.where(pretax > 0, np.nan)
    tax_rate = tax_provision / pretax_valid
    # clamp (pandas .clip PRESERVES NaN; do NOT use manual min/max):
    return tax_rate.clip(MIN_TAX_RATE, MAX_TAX_RATE)


def _nopat(cf: CompanyFinancials) -> pd.Series:
    """NOPAT = EBIT * (1 - tax_rate). Strictly EBIT-based (no fallback)."""
    ebit = pd.Series(cf.ebit, dtype="float64")
    return ebit * (1 - _tax_rate(cf))


def roic(cf: CompanyFinancials) -> list[float]:
    """ROIC = NOPAT / Invested Capital, per period.

    NOPAT = EBIT * (1 - tax_rate); IC = total_debt + total_equity - cash
    (period-end values, MVP). tax_rate = tax_provision / pretax_income,
    clamped to [MIN_TAX_RATE, MAX_TAX_RATE].

    NaN behavior: if pretax_income <= 0 the effective rate is undefined ->
    the period is NaN; if ic <= 0 (negative/zero capital is meaningless) ->
    NaN; if any component (ebit, tax_rate, debt, equity, cash) is NaN the
    result is NaN (pandas propagates).
    """
    nopat = _nopat(cf)

    # PART 3 — Invested Capital (period-end, MVP)
    # TODO (invested capital upgrade): use average IC instead of period-end
    # ((opening + closing) / 2) — costs one period of data but is
    # theoretically correct.
    total_debt = pd.Series(cf.total_debt, dtype="float64")
    total_equity = pd.Series(cf.total_equity, dtype="float64")
    cash = pd.Series(cf.cash, dtype="float64")
    ic = total_debt + total_equity - cash
    # ic <= 0 -> NaN (negative/zero capital is meaningless):
    ic = ic.where(ic > 0, np.nan)

    # PART 4 — Combine
    roic = nopat / ic
    roic = roic.replace([np.inf, -np.inf], np.nan)
    return roic.tolist()


def reinvestment_rate(cf: CompanyFinancials) -> list[float]:
    """Reinvestment rate = (net capex + ΔWC) / NOPAT, per period (Damodaran).

    net_capex = capex - depreciation. The provider normalizes capex to
    POSITIVE and yfinance depreciation is POSITIVE; therefore net_capex may
    be NEGATIVE — and that is VALID (an asset-light company invests less
    than it depreciates). ΔWC enters with its own sign (no abs; the sign
    carries information). The result may also be NEGATIVE — also VALID
    (releasing capital).

    NaN behavior: if NOPAT <= 0 the denominator is meaningless -> the period
    is NaN. If any component (capex, depreciation, ΔWC, NOPAT) is NaN the
    result is NaN (propagation).

    Note: the single-year value is noisy due to ΔWC volatility; the real
    signal is in the multi-year trend/stability (Stage 2).
    """
    capex = pd.Series(cf.capex, dtype="float64")
    depreciation = pd.Series(cf.depreciation, dtype="float64")
    change_in_working_capital = pd.Series(cf.change_in_working_capital, dtype="float64")

    nopat = _nopat(cf)

    net_capex = capex - depreciation
    numerator = net_capex + change_in_working_capital
    # NOPAT <= 0 periods become NaN (denominator meaningless):
    nopat_valid = nopat.where(nopat > 0, np.nan)

    reinvestment = numerator / nopat_valid
    reinvestment = reinvestment.replace([np.inf, -np.inf], np.nan)
    return reinvestment.tolist()


def rnd_to_revenue(cf: CompanyFinancials) -> list[float]:
    """R&D intensity: rnd_expense / revenue, per period.

    In sectors that do not report R&D (banks, energy) rnd_expense is NaN
    and the result is NaN — which is the CORRECT behavior (no imputation);
    sector-conditional use is handled in Stage 2. If revenue is NaN the
    period is NaN; if revenue == 0 it is NaN.
    """
    revenue = pd.Series(cf.revenue, dtype="float64")
    rnd_expense = pd.Series(cf.rnd_expense, dtype="float64")
    revenue_nonzero = revenue.replace(0, np.nan)
    ratio = rnd_expense / revenue_nonzero
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    return ratio.tolist()
