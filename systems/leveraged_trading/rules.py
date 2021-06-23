"""
Trading rules for Leveraged Trading system
"""
import pandas as pd


def smac(price, fast=16, slow=64):
    """
    Calculate the smac (Simple Moving Average Crossover) trading fule forecast, given a price and SMA speeds fast, slow

    Assumes that 'price' is daily data

    :param price: The price or other series to use (assumed Tx1)
    :type price: pd.Series

    :param fast: Lookback for fast in days
    :type fast: int

    :param slow: Lookback for slow in days
    :type slow: int

    :returns: pd.Series unscaled, uncapped forecast
    """

    fast_sma = price.ffill().rolling(window=fast).mean()
    slow_sma = price.ffill().rolling(window=slow).mean()
    raw_smac = fast_sma - slow_sma

    return raw_smac