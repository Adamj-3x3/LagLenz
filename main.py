import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt  # If you plan to visualize
import scipy.stats as stats

def get_stock_data(ticker, period="1y"):
    """
    Downloads historical stock data for a given ticker.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL").
        period (str): The time period for the data (e.g., "1y" for 1 year).

    Returns:
        pandas.Series: A Pandas Series containing the closing prices.
    """
    data = yf.download(ticker, period=period)
    return data["Close"]

ticker1 = "AAPL"
ticker2 = "MSFT"

data1 = get_stock_data(ticker1)
data2 = get_stock_data(ticker2)

print(f"Data for {ticker1}:")
print(data1.head())  # Print the first few rows
print("\n")
print(f"Data for {ticker2}:")
print(data2.head())