import math


def graham_number(stock_price, earnings_per_share, book_value_per_share):
    """
    Calculates the Graham Number for a given stock.

    Parameters:
    stock_price (float): The current price per share of the stock.
    earnings_per_share (float): The earnings per share (EPS) for the company.
    book_value_per_share (float): The book value per share (BVPS) for the company.

    Returns:
    float: The Graham Number for the stock.
    """
    multiplier = 22.5  # Graham's multiplier

    # Calculate the Graham Number
    graham_number = round(
        math.sqrt(multiplier * earnings_per_share * book_value_per_share), 2)

    print(graham_number)

    return graham_number


# Sample Data
stock_price = 50.00
earnings_per_share = 3.00
book_value_per_share = 25.00

graham_number(stock_price, earnings_per_share, book_value_per_share)
