"""Provider that implements the DataProvider contract via yfinance."""

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from .base import CompanyFinancials, DataProvider

logger = logging.getLogger(__name__)


INCOME_MAP = {
    "revenue": "Total Revenue",
    "cogs": "Cost Of Revenue",
    "operating_income": "Operating Income",
    "ebit": "EBIT",
    "pretax_income": "Pretax Income",
    "tax_provision": "Tax Provision",
    "rnd_expense": "Research And Development",
}

BALANCE_MAP = {
    "total_debt": "Total Debt",
    "total_equity": "Stockholders Equity",
    "cash": "Cash And Cash Equivalents",
}

CASHFLOW_MAP = {
    "operating_cash_flow": "Operating Cash Flow",
    "capex": "Capital Expenditure",
    "depreciation": "Depreciation And Amortization",
    "change_in_working_capital": "Change In Working Capital",
}


class YFinanceProvider(DataProvider):
    def get_financials(self, ticker: str) -> CompanyFinancials:
        tk = yf.Ticker(ticker)
        income = tk.financials
        balance = tk.balance_sheet
        cashflow = tk.cashflow

        date_index = (
            income.columns.union(balance.columns).union(cashflow.columns).sort_values()
        )

        if len(date_index) == 0:
            logger.warning(f"{ticker}: no table contains data (ticker may be invalid)")
            return CompanyFinancials(
                ticker=ticker,
                period_end_dates=[],
                revenue=[], cogs=[], operating_income=[], ebit=[],
                pretax_income=[], tax_provision=[], rnd_expense=[],
                depreciation=[], total_debt=[], total_equity=[], cash=[],
                operating_cash_flow=[], capex=[], change_in_working_capital=[],
            )

        def _extract(df: pd.DataFrame, yf_row_name: str, date_axis: pd.Index) -> list[float]:
            if yf_row_name not in df.index:
                logger.warning(f"{ticker}: '{yf_row_name}' row missing")
                return [np.nan] * len(date_axis)
            row = df.loc[yf_row_name]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            row = pd.to_numeric(row, errors="coerce")
            return row.reindex(date_axis).tolist()

        income_values = {k: _extract(income, v, date_index) for k, v in INCOME_MAP.items()}
        balance_values = {k: _extract(balance, v, date_index) for k, v in BALANCE_MAP.items()}
        cashflow_values = {k: _extract(cashflow, v, date_index) for k, v in CASHFLOW_MAP.items()}

        # The provider may standardize cross-source encoding of the same real-world
        # quantity (capex sign: yfinance returns negative, the engine expects
        # positive), but it must not modify/derive/fill data.
        # Convert capex to the positive-spend convention; NaNs stay NaN.
        cashflow_values["capex"] = [abs(v) for v in cashflow_values["capex"]]

        return CompanyFinancials(
            ticker=ticker,
            period_end_dates=list(date_index),
            **income_values,
            **balance_values,
            **cashflow_values,
        )
