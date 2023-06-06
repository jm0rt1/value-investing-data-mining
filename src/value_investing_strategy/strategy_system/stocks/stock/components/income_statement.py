

from dataclasses import dataclass
from pathlib import Path
from typing import List,  Union
from src.value_investing_strategy.strategy_system.stocks.stock.components.stock_component import StockComponent


from dataclasses import dataclass
from src.value_investing_strategy.strategy_system.stocks.stock.components.stock_component import StockComponent


@dataclass
class IncomeReport():
    fiscal_date_ending: str
    reported_currency: str
    gross_profit: float
    total_revenue: float
    cost_of_revenue: float
    cost_of_goods_and_services_sold: float
    operating_income: float
    selling_general_and_administrative: float
    research_and_development: float
    operating_expenses: float
    investment_income_net: float
    net_interest_income: float
    interest_income: float
    interest_expense: float
    depreciation: float
    depreciation_and_amortization: float
    income_before_tax: float
    income_tax_expense: float
    interest_and_debt_expense: float
    net_income_from_continuing_operations: float
    comprehensive_income_net_of_tax: float
    ebit: float
    ebitda: float
    net_income: float
    non_interest_income: float
    other_non_operating_income: float

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> 'IncomeReport':

        try:
            fiscal_date_ending = data.get("fiscalDateEnding", "")
        except ValueError as _:
            fiscal_date_ending = ""
        try:
            reported_currency = data.get("reportedCurrency", "")
        except ValueError as _:
            reported_currency = ""
        try:
            gross_profit = float(data.get("grossProfit", 0))
        except ValueError as _:
            gross_profit = 0
        try:
            total_revenue = float(data.get("totalRevenue", 0))
        except ValueError as _:
            total_revenue = 0
        try:
            cost_of_revenue = float(data.get("costOfRevenue", 0))
        except ValueError as _:
            cost_of_revenue = 0
        try:
            cost_of_goods_and_services_sold = float(
                data.get("costofGoodsAndServicesSold", 0))
        except ValueError as _:
            cost_of_goods_and_services_sold = 0
        try:
            operating_income = float(data.get("operatingIncome", 0))
        except ValueError as _:
            operating_income = 0
        try:
            selling_general_and_administrative = float(
                data.get("sellingGeneralAndAdministrative", 0))
        except ValueError as _:
            selling_general_and_administrative = 0
        try:
            operating_expenses = float(data.get("operatingExpenses", 0))
        except ValueError as _:
            operating_expenses = 0
        try:
            interest_expense = float(data.get("interestExpense", 0))
        except ValueError as _:
            interest_expense = 0
        try:
            depreciation_and_amortization = float(
                data.get("depreciationAndAmortization", 0))
        except ValueError as _:
            depreciation_and_amortization = 0
        try:
            income_before_tax = float(data.get("incomeBeforeTax", 0))
        except ValueError as _:
            income_before_tax = 0
        try:
            income_tax_expense = float(data.get("incomeTaxExpense", 0))
        except ValueError as _:
            income_tax_expense = 0
        try:
            net_income_from_continuing_operations = float(
                data.get("netIncomeFromContinuingOperations", 0))
        except ValueError as _:
            net_income_from_continuing_operations = 0
        try:
            comprehensive_income_net_of_tax = float(
                data.get("comprehensiveIncomeNetOfTax", 0))
        except ValueError as _:
            comprehensive_income_net_of_tax = 0
        try:
            ebit = float(data.get("ebit", 0))
        except ValueError as _:
            ebit = 0
        try:
            ebitda = float(data.get("ebitda", 0))
        except ValueError as _:
            ebitda = 0
        try:
            net_income = float(data.get("netIncome", 0))
        except ValueError as _:
            net_income = 0
        try:
            depreciation = float(data.get("depreciation", 0))
        except ValueError as _:
            depreciation = 0
        try:
            net_interest_income = float(data.get("netInterestIncome", 0))
        except ValueError as _:
            net_interest_income = 0
        try:
            non_interest_income = float(data.get("nonInterestIncome", 0))
        except ValueError as _:
            non_interest_income = 0
        try:
            interest_and_debt_expense = float(
                data.get("interestAndDebtExpense", 0))
        except ValueError as _:
            interest_and_debt_expense = 0
        try:
            other_non_operating_income = float(
                data.get("otherNonOperatingIncome", 0))
        except ValueError as _:
            other_non_operating_income = 0
        try:
            investment_income_net = float(data.get("investmentIncomeNet", 0))
        except ValueError as _:
            investment_income_net = 0

        try:
            interest_income = float(data.get("interestIncome", 0))
        except ValueError as _:
            interest_income = 0
        try:
            research_and_development = float(
                data.get("researchAndDevelopment", 0))
        except ValueError as _:
            research_and_development = 0

        return cls(
            fiscal_date_ending=fiscal_date_ending,
            reported_currency=reported_currency,
            gross_profit=gross_profit,
            total_revenue=total_revenue,
            cost_of_revenue=cost_of_revenue,
            cost_of_goods_and_services_sold=cost_of_goods_and_services_sold,
            operating_income=operating_income,
            selling_general_and_administrative=selling_general_and_administrative,
            research_and_development=research_and_development,
            operating_expenses=operating_expenses,
            investment_income_net=investment_income_net,
            net_interest_income=net_interest_income,
            interest_income=interest_income,
            interest_expense=interest_expense,
            depreciation=depreciation,
            depreciation_and_amortization=depreciation_and_amortization,
            income_before_tax=income_before_tax,
            income_tax_expense=income_tax_expense,
            interest_and_debt_expense=interest_and_debt_expense,
            net_income_from_continuing_operations=net_income_from_continuing_operations,
            comprehensive_income_net_of_tax=comprehensive_income_net_of_tax,
            ebit=ebit,
            ebitda=ebitda,
            net_income=net_income,
            non_interest_income=non_interest_income,
            other_non_operating_income=other_non_operating_income)


@ dataclass
class IncomeStatement(StockComponent):
    symbol: str
    annual_reports: List[IncomeReport]
    quarterly_reports: List[IncomeReport]

    @ classmethod
    def from_dict(cls, data: dict[str, Union[str, list[dict[str, str]]]]) -> 'IncomeStatement':
        annual_reports: list[IncomeReport] = []
        quarterly_reports: list[IncomeReport] = []

        raw_annual_reports: list[dict[str, str]
                                 ] = data.get("annualReports", [])  # type:ignore

        raw_qtrly_reports: list[dict[str, str]
                                ] = data.get("quarterlyReports", [])  # type:ignore

        if type(raw_annual_reports) is str:
            raise TypeError("Annual reports should not be a string")
        if type(raw_qtrly_reports) is str:
            raise TypeError("Quarterly reports should not be a string")

        for report in raw_annual_reports:
            annual_reports.append(IncomeReport.from_dict(report))
        for report in raw_qtrly_reports:
            quarterly_reports.append(IncomeReport.from_dict(report))

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
