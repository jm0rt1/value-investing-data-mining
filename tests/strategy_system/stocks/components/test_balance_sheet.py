from pathlib import Path
import unittest

from src.value_investing_strategy.strategy_system.stocks.stock.components.balance_sheet import BalanceReport, BalanceSheet

TEST_PATH = Path(
    "tests/test_files/inputs/strategy_system/stocks/stock/components/AAPL.BalanceSheet.json")
DATA = {
    "symbol": "AAPL",
    "annualReports": [
        {
            "fiscalDateEnding": "2022-09-30",
            "reportedCurrency": "USD",
            "totalAssets": "352755000000",
            "totalCurrentAssets": "135405000000",
            "cashAndCashEquivalentsAtCarryingValue": "23646000000",
            "cashAndShortTermInvestments": "48304000000",
            "inventory": "4946000000",
            "currentNetReceivables": "60932000000",
            "totalNonCurrentAssets": "217350000000",
            "propertyPlantEquipment": "42117000000",
            "accumulatedDepreciationAmortizationPPE": "72340000000",
            "intangibleAssets": "None",
            "intangibleAssetsExcludingGoodwill": "None",
            "goodwill": "None",
            "investments": "292870000000",
            "longTermInvestments": "120805000000",
            "shortTermInvestments": "24658000000",
            "otherCurrentAssets": "21223000000",
            "otherNonCurrentAssets": "54428000000",
            "totalLiabilities": "302083000000",
            "totalCurrentLiabilities": "153982000000",
            "currentAccountsPayable": "64115000000",
            "deferredRevenue": "20312000000",
            "currentDebt": "21239000000",
            "shortTermDebt": "9982000000",
            "totalNonCurrentLiabilities": "148101000000",
            "capitalLeaseObligations": "812000000",
            "longTermDebt": "110100000000",
            "currentLongTermDebt": "11128000000",
            "longTermDebtNoncurrent": "98959000000",
            "shortLongTermDebtTotal": "233256000000",
            "otherCurrentLiabilities": "60845000000",
            "otherNonCurrentLiabilities": "49142000000",
            "totalShareholderEquity": "50672000000",
            "treasuryStock": "None",
            "retainedEarnings": "-3068000000",
            "commonStock": "64849000000",
            "commonStockSharesOutstanding": "15943425000"
        }
    ],
    "quarterlyReports": [
        {
            "fiscalDateEnding": "2022-12-31",
            "reportedCurrency": "USD",
            "totalAssets": "346747000000",
            "totalCurrentAssets": "128777000000",
            "cashAndCashEquivalentsAtCarryingValue": "20535000000",
            "cashAndShortTermInvestments": "51355000000",
            "inventory": "6820000000",
            "currentNetReceivables": "54180000000",
            "totalNonCurrentAssets": "217970000000",
            "propertyPlantEquipment": "42951000000",
            "accumulatedDepreciationAmortizationPPE": "68044000000",
            "intangibleAssets": "None",
            "intangibleAssetsExcludingGoodwill": "None",
            "goodwill": "None",
            "investments": "291347000000",
            "longTermInvestments": "114095000000",
            "shortTermInvestments": "30820000000",
            "otherCurrentAssets": "16422000000",
            "otherNonCurrentAssets": "60924000000",
            "totalLiabilities": "290020000000",
            "totalCurrentLiabilities": "137286000000",
            "currentAccountsPayable": "57918000000",
            "deferredRevenue": "20592000000",
            "currentDebt": "11483000000",
            "shortTermDebt": "1743000000",
            "totalNonCurrentLiabilities": "152734000000",
            "capitalLeaseObligations": "None",
            "longTermDebt": "109400000000",
            "currentLongTermDebt": "9740000000",
            "longTermDebtNoncurrent": "99627000000",
            "shortLongTermDebtTotal": "111143000000",
            "otherCurrentLiabilities": "59893000000",
            "otherNonCurrentLiabilities": "53107000000",
            "totalShareholderEquity": "56727000000",
            "treasuryStock": "None",
            "retainedEarnings": "3240000000",
            "commonStock": "66399000000",
            "commonStockSharesOutstanding": "15842407000"
        }
    ]
}


class TestBalanceReport(unittest.TestCase):
    def test_load_from_dict(self,):
        dict_report: dict[str, str] = DATA["quarterlyReports"][0]  # nopep8  #type:ignore
        report = BalanceReport.from_dict(dict_report)

        self.assertIsInstance(report, BalanceReport)
        self.assertEqual(report.fiscal_date_ending, "2022-12-31")
        self.assertEqual(report.reported_currency, "USD")
        self.assertEqual(report.total_assets, 346747000000)
        self.assertEqual(report.total_current_assets, 128777000000)
        self.assertEqual(
            report.cash_and_cash_equivalents_at_carrying_value, 20535000000)
        self.assertEqual(
            report.cash_and_short_term_investments, 51355000000)
        self.assertEqual(report.inventory, 6820000000)
        self.assertEqual(report.current_net_receivables, 54180000000)
        self.assertEqual(report.total_non_current_assets, 217970000000)
        self.assertEqual(report.property_plant_equipment, 42951000000)
        self.assertEqual(
            report.accumulated_depreciation_amortization_ppe, 68044000000)
        self.assertEqual(report.intangible_assets, 0)
        self.assertEqual(report.intangible_assets_excluding_goodwill, 0)
        self.assertEqual(report.goodwill, 0)
        self.assertEqual(report.investments, 291347000000)
        self.assertEqual(report.long_term_investments, 114095000000)
        self.assertEqual(report.short_term_investments, 30820000000)
        self.assertEqual(report.other_current_assets, 16422000000)
        self.assertEqual(report.other_non_current_assets, 60924000000)
        self.assertEqual(report.total_liabilities, 290020000000)
        self.assertEqual(report.total_current_liabilities, 137286000000)
        self.assertEqual(report.current_accounts_payable, 57918000000)
        self.assertEqual(report.deferred_revenue, 20592000000)
        self.assertEqual(report.current_debt, 11483000000)
        self.assertEqual(report.short_term_debt, 1743000000)
        self.assertEqual(report.total_non_current_liabilities, 152734000000)
        self.assertEqual(report.capital_lease_obligations, 0)
        self.assertEqual(report.long_term_debt, 109400000000)
        self.assertEqual(report.current_long_term_debt, 9740000000)
        self.assertEqual(report.long_term_debt_noncurrent, 99627000000)
        self.assertEqual(report.short_long_term_debt_total, 111143000000)
        self.assertEqual(report.other_current_liabilities, 59893000000)
        self.assertEqual(report.other_non_current_liabilities, 53107000000)
        self.assertEqual(report.total_shareholder_equity, 56727000000)
        self.assertEqual(report.treasury_stock, 0)
        self.assertEqual(report.retained_earnings, 3240000000)
        self.assertEqual(report.common_stock, 66399000000)
        self.assertEqual(
            report.common_stock_shares_outstanding, 15842407000)


class TestBalanceStatement(unittest.TestCase):
    def test_from_dict(self):
        statement = BalanceSheet.from_dict(DATA)
        self.assertIsInstance(statement, BalanceSheet)
        self.assertEqual(len(statement.annual_reports), 1)
        self.assertIsInstance(statement.annual_reports[0], BalanceReport)
        self.assertEqual(len(statement.quarterly_reports), 1)
        self.assertIsInstance(statement.quarterly_reports[0], BalanceReport)

    def test_from_file(self):
        statement = BalanceSheet.from_json_file(TEST_PATH)
        self.assertIsInstance(statement, BalanceSheet)
        self.assertEqual(len(statement.annual_reports), 5)
        self.assertIsInstance(statement.annual_reports[0], BalanceReport)
        self.assertEqual(len(statement.quarterly_reports), 20)
        self.assertIsInstance(statement.quarterly_reports[0], BalanceReport)
