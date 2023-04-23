

from dataclasses import dataclass
from typing import Union
from value_investing_strategy.strategy_system.stocks.stock.components.StockComponent import StockComponent
from pathlib import Path


@dataclass
class BalanceReport():
    fiscal_date_ending: str
    reported_currency: str
    total_assets: float
    total_current_assets: float
    cash_and_cash_equivalents_at_carrying_value: float
    cash_and_short_term_investments: float
    inventory: float
    current_net_receivables: float
    total_non_current_assets: float
    property_plant_equipment: float
    accumulated_depreciation_amortization_ppe: float
    intangible_assets: float
    intangible_assets_excluding_goodwill: float
    goodwill: float
    investments: float
    long_term_investments: float
    short_term_investments: float
    other_current_assets: float
    other_non_current_assets: float
    total_liabilities: float
    total_current_liabilities: float
    current_accounts_payable: float
    deferred_revenue: float
    current_debt: float
    short_term_debt: float
    total_non_current_liabilities: float
    capital_lease_obligations: float
    long_term_debt: float
    current_long_term_debt: float
    long_term_debt_noncurrent: float
    short_long_term_debt_total: float
    other_current_liabilities: float
    other_non_current_liabilities: float
    total_shareholder_equity: float
    treasury_stock: float
    retained_earnings: float
    common_stock: float
    common_stock_shares_outstanding: float

    @classmethod
    def from_dict(cls, data: dict[str, str]):
        try:
            intangible_assets = float(data.get(
                "proceedsFromOperatingActivities", ""))
        except ValueError as _:
            intangible_assets = 0
        try:
            intangible_assets_excluding_goodwill = float(data.get(
                "proceedsFromOperatingActivities", ""))
        except ValueError as _:
            intangible_assets_excluding_goodwill = 0
        try:
            goodwill = float(data.get(
                "proceedsFromOperatingActivities", ""))
        except ValueError as _:
            goodwill = 0
        try:
            capital_lease_obligations = float(data.get(
                "proceedsFromOperatingActivities", ""))
        except ValueError as _:
            capital_lease_obligations = 0

        try:
            treasury_stock = float(data.get(
                "proceedsFromOperatingActivities", ""))
        except ValueError as _:
            treasury_stock = 0

        try:
            deferred_revenue = float(data.get(
                "deferredRevenue", ""))
        except ValueError as _:
            deferred_revenue = 0

        try:
            inventory = float(data.get(
                "inventory", ""))
        except ValueError as _:
            inventory = 0
        try:
            current_net_receivables = float(
                data.get("currentNetReceivables", ""))
        except ValueError as _:
            current_net_receivables = 0

        try:
            other_non_current_assets = float(
                data.get("otherNonCurrentAssets", ""))
        except ValueError as _:
            other_non_current_assets = 0

        try:
            current_accounts_payable = float(
                data.get("currentAccountsPayable", ""))
        except ValueError as _:
            current_accounts_payable = 0

        try:
            other_non_current_assets = float(
                data.get("otherNonCurrentAssets", ""))
        except ValueError as _:
            other_non_current_assets = 0

        try:
            other_non_current_assets = float(
                data.get("otherNonCurrentAssets", ""))
        except ValueError as _:
            other_non_current_assets = 0
        try:
            investments = float(data.get("investments", ""))
        except ValueError as _:
            investments = 0

        try:
            long_term_investments = float(data.get("longTermInvestments", ""))
        except ValueError as _:
            long_term_investments = 0

        try:
            short_term_investments = float(
                data.get("shortTermInvestments", ""))
        except ValueError as _:
            short_term_investments = 0

        try:
            accumulated_depreciation_amortization_ppe = float(data.get(
                "accumulatedDepreciationAmortizationPPE", ""))
        except ValueError as _:
            accumulated_depreciation_amortization_ppe = 0

        try:
            current_debt = float(data.get("currentDebt", ""))
        except ValueError as _:
            current_debt = 0

        try:
            short_term_debt = float(data.get("shortTermDebt", ""))
        except ValueError as _:
            short_term_debt = 0
        try:
            fiscal_date_ending = data.get("fiscalDateEnding", "")
        except ValueError as _:
            fiscal_date_ending = ""
        try:
            reported_currency = data.get("reportedCurrency", "")
        except ValueError as _:
            reported_currency = ""
        try:
            total_assets = float(data.get("totalAssets", ""))
        except ValueError as _:
            total_assets = 0
        try:
            total_current_assets = float(data.get("totalCurrentAssets", ""))
        except ValueError as _:
            total_current_assets = 0
        try:
            cash_and_cash_equivalents_at_carrying_value = float(data.get(
                "cashAndCashEquivalentsAtCarryingValue", ""))
        except ValueError as _:
            cash_and_cash_equivalents_at_carrying_value = 0
        try:
            cash_and_short_term_investments = float(data.get(
                "cashAndShortTermInvestments", ""))
        except ValueError as _:
            cash_and_short_term_investments = 0
        try:
            total_non_current_assets = float(
                data.get("totalNonCurrentAssets", ""))
        except ValueError as _:
            total_non_current_assets = 0
        try:
            property_plant_equipment = float(
                data.get("propertyPlantEquipment", ""))
        except ValueError as _:
            property_plant_equipment = 0
        try:
            other_current_assets = float(data.get("otherCurrentAssets", ""))
        except ValueError as _:
            other_current_assets = 0
        try:
            total_liabilities = float(data.get("totalLiabilities", ""))
        except ValueError as _:
            total_liabilities = 0
        try:
            total_current_liabilities = float(
                data.get("totalCurrentLiabilities", ""))
        except ValueError as _:
            total_current_liabilities = 0
        try:
            total_non_current_liabilities = float(data.get(
                "totalNonCurrentLiabilities", ""))
        except ValueError as _:
            total_non_current_liabilities = 0
        try:
            long_term_debt = float(data.get("longTermDebt", ""))
        except ValueError as _:
            long_term_debt = 0
        try:
            current_long_term_debt = float(data.get("currentLongTermDebt", ""))
        except ValueError as _:
            current_long_term_debt = 0
        try:
            long_term_debt_noncurrent = float(
                data.get("longTermDebtNoncurrent", ""))
        except ValueError as _:
            long_term_debt_noncurrent = 0
        try:
            short_long_term_debt_total = float(
                data.get("shortLongTermDebtTotal", ""))
        except ValueError as _:
            short_long_term_debt_total = 0
        try:
            other_current_liabilities = float(
                data.get("otherCurrentLiabilities", ""))
        except ValueError as _:
            other_current_liabilities = 0
        try:
            other_non_current_liabilities = float(data.get(
                "otherNonCurrentLiabilities", ""))
        except ValueError as _:
            other_non_current_liabilities = 0
        try:
            total_shareholder_equity = float(
                data.get("totalShareholderEquity", ""))
        except ValueError as _:
            total_shareholder_equity = 0
        try:
            retained_earnings = float(data.get("retainedEarnings", ""))
        except ValueError as _:
            retained_earnings = 0
        try:
            common_stock = float(data.get("commonStock", ""))

        except ValueError as _:
            common_stock = 0
        try:
            common_stock_shares_outstanding = float(data.get(
                "commonStockSharesOutstanding", ""))
        except ValueError as _:
            common_stock_shares_outstanding = 0
        return cls(
            fiscal_date_ending=fiscal_date_ending,
            reported_currency=reported_currency,
            total_assets=total_assets,
            total_current_assets=total_current_assets,
            cash_and_cash_equivalents_at_carrying_value=cash_and_cash_equivalents_at_carrying_value,
            cash_and_short_term_investments=cash_and_short_term_investments,
            inventory=inventory,
            current_net_receivables=current_net_receivables,
            total_non_current_assets=total_non_current_assets,
            property_plant_equipment=property_plant_equipment,
            accumulated_depreciation_amortization_ppe=accumulated_depreciation_amortization_ppe,
            intangible_assets=intangible_assets,
            intangible_assets_excluding_goodwill=intangible_assets_excluding_goodwill,
            goodwill=goodwill,
            investments=investments,
            long_term_investments=long_term_investments,
            short_term_investments=short_term_investments,
            other_current_assets=other_current_assets,
            other_non_current_assets=other_non_current_assets,
            total_liabilities=total_liabilities,
            total_current_liabilities=total_current_liabilities,
            current_accounts_payable=current_accounts_payable,
            deferred_revenue=deferred_revenue,
            current_debt=current_debt,
            short_term_debt=short_term_debt,
            total_non_current_liabilities=total_non_current_liabilities,
            capital_lease_obligations=capital_lease_obligations,
            long_term_debt=long_term_debt,
            current_long_term_debt=current_long_term_debt,
            long_term_debt_noncurrent=long_term_debt_noncurrent,
            short_long_term_debt_total=short_long_term_debt_total,
            other_current_liabilities=other_current_liabilities,
            other_non_current_liabilities=other_non_current_liabilities,
            total_shareholder_equity=total_shareholder_equity,
            treasury_stock=treasury_stock,
            retained_earnings=retained_earnings,
            common_stock=common_stock,
            common_stock_shares_outstanding=common_stock_shares_outstanding)


@dataclass
class BalanceSheet(StockComponent):
    symbol: str
    annual_reports: list[BalanceReport]
    quarterly_reports: list[BalanceReport]

    @ classmethod
    def from_dict(cls, data: dict[str, Union[str, list[dict[str, str]]]]):
        annual_reports: list[BalanceReport] = []
        quarterly_reports: list[BalanceReport] = []

        raw_annual_reports: list[dict[str, str]
                                 ] = data.get("annualReports", [])  # type:ignore

        raw_qtrly_reports: list[dict[str, str]
                                ] = data.get("quarterlyReports", [])  # type:ignore

        if type(raw_annual_reports) is str:
            raise TypeError("Annual reports should not be a string")
        if type(raw_qtrly_reports) is str:
            raise TypeError("Quarterly reports should not be a string")

        for report in raw_annual_reports:
            annual_reports.append(BalanceReport.from_dict(report))
        for report in raw_qtrly_reports:
            quarterly_reports.append(BalanceReport.from_dict(report))

        symbol = data.get("symbol", "")
        if type(symbol) is not str:
            raise TypeError("symbol needs to be a string")
        return cls(
            symbol=symbol,
            annual_reports=annual_reports,
            quarterly_reports=quarterly_reports
        )

    @classmethod
    def from_json_file(cls, path: Path):
        return cls.from_dict(cls.load_json_dict(path))
