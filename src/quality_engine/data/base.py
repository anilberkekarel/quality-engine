"""Veri katmanının soyut sözleşmesi: ham finansal taşıyıcı ve sağlayıcı arayüzü."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CompanyFinancials:
    ticker: str
    period_end_dates: list
    revenue: list[float]
    cogs: list[float]
    operating_income: list[float]
    pretax_income: list[float]
    tax_provision: list[float]
    rnd_expense: list[float]
    depreciation: list[float]
    total_debt: list[float]
    total_equity: list[float]
    cash: list[float]
    operating_cash_flow: list[float]
    capex: list[float]
    change_in_working_capital: list[float]


class DataProvider(ABC):
    @abstractmethod
    def get_financials(self, ticker: str) -> CompanyFinancials:
        """Verilen ticker için ham finansalları döndüren sözleşme."""
        ...
