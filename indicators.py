import numpy as np
import pandas as pd


def SMA(data: pd.DataFrame, period: int =30, offset: int =0, column: str ='Close') -> pd.Series:
    '''
    Simple Moving Average (SMA)
    data -> DataFrame
    period -> int: The period from which SMA should calculate the average price
    column -> str: Open / Close / High / Low : Which column of DataFrame to use for generating the moving average
    offset -> int: The amount we want to shift the SMA

    Formula: SMA[n+period] = (Day[n].Price + Day[n+1].Price + ... + Day[n+period].Price) / period
    '''
    # pandas has built-in function for rolling windows which is: pandas.DataFrame.rolling
    #   window parameter: is equal to period that we're looking for
    #       Since we need average of this rolling window, we use mean() function for the rolling window
    roll = data[column].rolling(window=period).mean()
    return roll.shift(periods=offset)


def EMA(data: pd.DataFrame, period: int =20, offset: int =0, column: str ='Close') -> pd.Series:
    '''
    Exponential Moving Average (EMA) / Exponential Weighted Moving Average
    data -> DataFrame
    period -> int: The period from which EMA should calculate the average price
    column -> str: Open / Close / High / Low : Which column of DataFrame to use for generating the moving average
    offset -> int: The amount we want to shift the EMA

    Formula: EMA[today] = (Day[today].Price * multiplier) + (EMA[yesterday] * (1-multiplier))
                multiplier = Smoothing / (1 + period)
                    There are many possible choices for the smoothing factor, the most common choice is 2
    '''
    # pandas has built-in exponential weighted function which is: pandas.DataFrame.ewm
    #   span parameter: is equal to period that we're looking for
    #   adjust parameter: if it's flase, the function calculate the result recursively
    ema = data[column].ewm(span=period, adjust=False).mean()
    return ema.shift(periods=offset)


def MACD(data: pd.DataFrame, period_long: int =26, period_short: int =12, period_signal: int =9, column: str ='Close') -> (pd.Series, pd.Series):
    '''
    Moving Average Convergence Divergence (MACD)
    data -> DataFrame
    period_long -> int: The period for long EMA
    period_short -> int: The period for short EMA
    period_signal -> int: The period for MACD
    column -> str: Open / Close / High / Low : Which column of DataFrame to use for generating the moving average

    Formula: MACD[n] = EMA_short[n] - EMA_long[n]
            Signal[n] = EMA(MACD)[n]
    '''
    ShortEMA = EMA(data, period_short, column=column)
    LongEMA = EMA(data, period_long, column=column)
    # Return desired outputs as (MACD, Signal)
    return (ShortEMA - LongEMA, EMA(pd.DataFrame({'MACD': (ShortEMA - LongEMA)}), period_signal, column='MACD'))


def RSI(data: pd.DataFrame, period: int =14, column: str ='Close') -> pd.Series:
    '''
    Relative Strength Index (RSI)
    data -> DataFrame
    period -> int: The look-back period
    column -> str: Open / Close / High / Low : Which column of DataFrame to use for generating the moving average

    Formula: 100 - (100 / (1 + RS))
                RS: Average Gain / Average Loss
                    First Avg Gain: sum(Gains over the past period) / period
                    First Avg Loss: sum(Losses over the past period) / period

                    Avg Gain[n]: (Avg Gain[n-1] * (period-1) + Gain[n]) / period
                    Avg Loss[n]: (Avg Loss[n-1] * (period-1) + Loss[n]) / period

                    Change[n] = Close[n] - Close[n-1]
                    Gain[n] = Change[n] if Change[n] > 0 else Loss[n] = Change[n]

    '''
    # pandas has built-in functin to calculate a difference for each row or column in DataFrame: pandas.DataFrame.diff
    # Finding the change of Close columns and store them in delta
    delta = data[column].diff(periods=1)
    delta = delta[1:] # Removing the first element of delta which is nan
    # Creating two series for Gain and Loss and change the unappropriate elements to 0
    gain = delta.copy()
    loss = delta.copy()
    gain[gain<0] = 0
    loss[loss>0] = 0
    tmp = pd.DataFrame({'Gain':gain, 'Loss':loss}) # Making a DataFrame using gain and loss
    # Calculating AvgGain and AvgLoss using SMA
    avgGain = SMA(tmp, period, column='Gain')
    avgLoss = abs(SMA(tmp, period, column='Loss')) # Since all the losses are less than zero, we habe to use abs() function
    del tmp
    RS = avgGain/avgLoss
    RSI = 100.0 - (100.0/(1.0 + RS))
    return RSI


def StochasticOscillator(data: pd.DataFrame, period_fast: int =14, period_slow: int =3, offset: int =3) -> (pd.Series, pd.Series):
    '''
    data -> DataFrame
    period_fast -> int: The period for fast stochastic indicator : %K (PK)
    period_slow -> int: The period for slow stochastic indicator : %D (PD)
    offset -> int: Smoothing parameter for %K

    Formula: PK = 100 * ((C - L_fast)/(H_fast - L_fast))
                C: The most recent closing price
                L_fast: The lowest price traded of the period_fast previous trading sessions
                H_fast: The highest price traded during the same period_fast previous trading sessions
            DK = SMA(PK, period_slow)
    '''
    # pandas has built-in function for rolling windows which is: pandas.DataFrame.rolling
    #   window parameter: is equal to period that we're looking for
    C = data['Close'].rolling(window=period_fast).apply(lambda x: x[-1]) # We need most recent closing price -> [-1] elemnt of period_fast
    L_fast = data['Low'].rolling(window=period_fast).min() # We need min() of Low column
    H_fast = data['High'].rolling(window=period_fast).max() # We need max() of High column
    percentageK = 100 * ((C - L_fast)/(H_fast - L_fast)) # Calculating PK
    percentageK = percentageK.to_frame() # Change PK to DataFrame 
    percentageK['PK'] = SMA(percentageK, period=offset, column=0) # Adding PK to DataFrame
    percentageK['PD'] = SMA(percentageK, period=period_slow, column='PK') # Calculating PD
    return (percentageK['PK'], percentageK['PD'])


def OBV(data) -> pd.Series:
    '''
    On-Balance Volume (OBV)
    data -> DataFrame

    Formula: OBV[today] = OBV[yesterday] + (
                                            Volume,   if Close[today] > Close[yesterday]
                                            0,        if Close[today] = Close[yesterday]
                                            -Volume,  if Close[today] < Close[yesterday]
                                            )
    '''
    # pandas has built-in functin to calculate a difference for each row or column in DataFrame: pandas.DataFrame.diff
    #   Using that function to find out wether 
    #       Close[today]-Close[yesterday] is greater, less than, or equal to 0 for further use.
    df = data['Close'].diff().to_frame(name='dif')
    df.iloc[0, :]['dif'] = 0 # Assigning the first element of a column to 0
    df['Vol'] = data['Volume']
    def obv(row):
        '''
        A function to calculate OBV for future use on pandas DataFrame
        '''
        if row['dif'] > 0:
            return row['Vol']
        elif row['dif'] < 0:
            return -row['Vol']
        else:
            return 0
    # Using obv function for the DataFrame and calculate the cumulative sum of the column
    return df.apply(lambda row: obv(row), axis=1).cumsum()