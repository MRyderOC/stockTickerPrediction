import numpy as np
import pandas as pd


def volumeCalculatorToDollars(data: pd.DataFrame) -> pd.Series:
    return data['Volume'] * data['Close']




def findGapsValue(data: pd.DataFrame) -> pd.Series:
    """
    Finding gaps' values between today's Open price and yesterday Close price.
        These gaps will happen for after-hour trading, fundamental impact, etc.
    """
    return data['Open'] - data['Close'].shift(1)


def findGapsPercentage(data: pd.DataFrame) -> pd.Series:
    """
    Finding gaps' percentage between today's Open price and yesterday Close price.
        These gaps will happen for after-hour trading, fundamental impact, etc.
    """
    return (findGapsValue(data) / data['Close'].shift(1)) * 100





def termReturnValue(data: pd.DataFrame) -> pd.Series:
    """
    Term here refers to the time frame the data has (daily, weekly, ...)
    """
    return data['Close'] - data['Open']


def termReturnPercentage(data: pd.DataFrame) -> pd.Series:
    """
    Term here refers to the time frame the data has (daily, weekly, ...)
    """
    return 100 * termReturnValue(data) / data['Open']





def valueReturn(data: pd.DataFrame, column: str ='Close') -> pd.Series:
    """
    Value between today's desired column and yesterdays
    """
    return data[column] - data[column].shift(1)


def valueReturnPercentage(data: pd.DataFrame, column: str ='Close') -> pd.Series:
    """
    Percentage of the value between today's desired column and yesterdays
    """
    return (valueReturn(data) / data[column].shift(1)) * 100





def wholeFluctuationValue(data: pd.DataFrame) -> pd.Series:
    return data['High'] - data['Low']


def wholeFluctuationPercentage(data: pd.DataFrame) -> pd.Series:
    return 100 * wholeFluctuationValue(data) / data['Open']


def originalFluctuationValue(data: pd.DataFrame) -> pd.Series:
    return abs(data['Open'] - data['Close'])


def originalFluctuationPercentage(data: pd.DataFrame) -> pd.Series:
    return 100 * originalFluctuationValue(data) / data['Open']


def shadowFluctuationValue(data: pd.DataFrame) -> pd.Series:
    return wholeFluctuationValue(data) - originalFluctuationValue(data)


def shadowFluctuationPercentage(data: pd.DataFrame) -> pd.Series:
    return 100 * shadowFluctuationValue(data) / data['Open']





def changePeriod(data: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Change a DataFrame with daily index to desired period
    Valid periods:
        W: Weekly
        M: Monthly
        3M: Quarterly
        Y: Yearly
    """
    df = pd.DataFrame()
    df['Close'] = data['Close'].resample(period).apply(lambda x: x[-1]) # Assign last Close
    df['Open'] = data['Open'].resample(period).apply(lambda x: x[0]) # Assign first Open
    df['Low'] = data['Low'].resample(period).min() # Assign Low for the desired period
    df['High'] = data['High'].resample(period).max() # Assign High for the desired period
    df['Volume'] = data['Volume'].resample(period).sum() # Assign sum of volumes for the desired period
    df['stockSplits'] = data['stockSplits'].resample(period).sum() # Assign sum of stockSplits for the desired period
    return df
