# Importing libraries
    # Basic libs
import sys
import numpy as np
import pandas as pd
    # Feature creation
import indicators
import featureCreator
    # API
import yfinance as yf
    # Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from fbprophet import Prophet


def getTicker() -> str:
    '''
    Get the ticker from user.
    '''
    return input('Please enter the desired ticker: ').upper()





def getHistoricalData(ticker: str) -> pd.DataFrame:
    '''
    Reading the historical data from yfinance library.
    '''
    Ticker = yf.Ticker(ticker) # Get the ticker using yfinance lib
    return Ticker.history(period='max') # Return ticker's historical data


def findLastSplit(data: pd.DataFrame) -> str:
    '''
    Finding the last date the ticker splited.
    '''
    data.stockSplits.replace(0.0, np.nan, inplace=True) # Clean the data
    splits = data[data.stockSplits.notnull()] # Find all splits
    if sum(splits.any()):
        last = splits.tail(1).index # Find the last split
        return last.astype(str)[0]
    else: # If there is no splits
        print('There has no splits for this ticker')
        return '1970-01-01'


def addingBasicFeatures(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Add some basic features to the DataFrame for further use
    '''
    data['gapValue'] = featureCreator.findGapsValue(data)
    data['gapPercentage'] = featureCreator.findGapsPercentage(data)
    data['valueReturn'] = featureCreator.valueReturn(data)
    data['valueReturnPercentage'] = featureCreator.valueReturnPercentage(data)
    data['termReturnValue'] = featureCreator.termReturnValue(data)
    data['termReturnPercentage'] = featureCreator.termReturnPercentage(data)
    data['volumeInDollars'] = featureCreator.volumeCalculatorToDollars(data)
    
    data['originalFluctuationPercentage'] = featureCreator.originalFluctuationPercentage(data)
    data['originalFluctuationValue'] = featureCreator.originalFluctuationValue(data)
    data['wholeFluctuationPercentage'] = featureCreator.wholeFluctuationPercentage(data)
    data['wholeFluctuationValue'] = featureCreator.wholeFluctuationValue(data)
    data['shadowFluctuationPercentage'] = featureCreator.shadowFluctuationPercentage(data)
    data['shadowFluctuationValue'] = featureCreator.shadowFluctuationValue(data)
    
    return data


def addingBasicIndicators(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Add some basic indicators to the DataFrame for further use
    '''
    data['SMA5'] = indicators.SMA(data, period=5)
    data['SMA15'] = indicators.SMA(data, period=15)
    data['SMA50'] = indicators.SMA(data, period=50)
    data['SMA100'] = indicators.SMA(data, period=100)
    data['SMA200'] = indicators.SMA(data, period=200)

    data['EMA5'] = indicators.EMA(data, period=5)
    data['EMA15'] = indicators.EMA(data, period=15)
    data['EMA50'] = indicators.EMA(data, period=50)
    data['EMA100'] = indicators.EMA(data, period=100)
    data['EMA200'] = indicators.EMA(data, period=200)

    data['RSI'] = indicators.RSI(data)
    data['OBV'] = indicators.OBV(data)

    data['MACD'], data['Signal'] = indicators.MACD(data)
    data['PK'], data['PD'] = indicators.StochasticOscillator(data)
    
    return data


def dataCleaning_addPredictionColumn(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Clean the data for further use.
    '''
    data['Pred'] = data.Open.shift(periods=-1) # Prediction column is tomorrow's open price
    return data


def dataCleaning_cleaned(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Clean the data for further use.
    '''
    df = data.drop(['stockSplits'], axis=1) # Drop the stockSplit column
    df.dropna(inplace=True) # Drop all the rows 
    return df


def dataPreparation(ticker: str, timeFrame: str = 'D') -> (pd.DataFrame, pd.DataFrame, str):
    '''
    Get the desired ticker historical data and put together all requirements for building the model.
    
    timeFrame -> str:
        Valid options:
            D: Daily
            W: Weekly
    '''
    # Get the data
    df = getHistoricalData(ticker)
    # Rename Stock splits columns
    df.rename(columns={'Stock Splits':'stockSplits'}, inplace=True)
    # Drop unnecessary Dividend column
    df.drop(['Dividends'], axis=1, inplace=True)
    # Check the desired time frame
    if timeFrame == 'W':
        df = featureCreator.changePeriod(df, period='W')
    # Find split date
    splitDate = findLastSplit(df)
    # Add features
    df = addingBasicFeatures(df)
    df = addingBasicIndicators(df)
    # Clean the data
    originalAndPredDF = dataCleaning_addPredictionColumn(df)
    cleanedDF = dataCleaning_cleaned(originalAndPredDF)

    return originalAndPredDF, cleanedDF, splitDate





def modelBuildingRF(originalAndPredData: pd.DataFrame, cleanedData: pd.DataFrame, split_date: str) -> float:
    '''Random Forest Model'''
    # Making y and X for further usage in the model
    yTomorrowRF = cleanedData.Pred.loc[split_date:]
    yTomorrowRF = yTomorrowRF.reset_index()
    yTomorrowRF.drop(['Date'], axis=1, inplace=True)
    XTomorrowRF = cleanedData.loc[split_date:]
    XTomorrowRF.reset_index(inplace=True)
    XTomorrowRF.drop(['Date', 'Pred'], axis=1, inplace=True)
    # Make today data as test data
    XTomorrowRF_test = originalAndPredData.tail(1).reset_index()
    XTomorrowRF_test.drop(['Date', 'Pred', 'stockSplits'], axis=1, inplace=True)
    # Making the model and fit it
    tomorrowModelRF = RandomForestRegressor(random_state=0)
    tomorrowModelRF.fit(XTomorrowRF, yTomorrowRF.values.ravel())
    # Predict tomorrow
    tomorrowPredictionRF = tomorrowModelRF.predict(XTomorrowRF_test)
    
    return tomorrowPredictionRF[0]


def modelBuildingProphet(originalAndPredData: pd.DataFrame, cleanedData: pd.DataFrame, split_date: str) -> (float, float, float):
    '''Prophet library Model'''
    # Making the dataframe compatible with Prophet library
    yProTomorrow = originalAndPredData.Open.loc[split_date:]
    yProTomorrow = yProTomorrow.reset_index()
    yProTomorrow.rename(columns={'Date':'ds', 'Open':'y'}, inplace=True)
    # Making the model and fit it
    tomorrowModel = Prophet()
    tomorrowModel.fit(yProTomorrow)
    # Make tomorrow for predictions
    tomorrowDate = tomorrowModel.make_future_dataframe(periods=1)
    # Predict tomorrow
    tomorrowPrediction = tomorrowModel.predict(tomorrowDate)

    ind = tomorrowPrediction.yhat.tail(1).index[0]
    return tomorrowPrediction.yhat.loc[ind], tomorrowPrediction.yhat_upper.loc[ind], tomorrowPrediction.yhat_lower.loc[ind]


def modelBuildingLR(originalAndPredData: pd.DataFrame, cleanedData: pd.DataFrame, split_date: str) -> float:
    '''Linear Regression Model'''
    # Making y and X for further usage in the model
    yTomorrowLR = cleanedData.Pred.loc[split_date:]
    yTomorrowLR = yTomorrowLR.reset_index()
    yTomorrowLR.drop(['Date'], axis=1, inplace=True)
    XTomorrowLR = cleanedData.loc[split_date:]
    XTomorrowLR.reset_index(inplace=True)
    XTomorrowLR.drop(['Date', 'Pred'], axis=1, inplace=True)
    # Make today data as test data
    XTomorrowLR_test = originalAndPredData.tail(1).reset_index()
    XTomorrowLR_test.drop(['Date', 'Pred', 'stockSplits'], axis=1, inplace=True)
    # Making the model and fit it
    tomorrowModelLR = LinearRegression(normalize=True, copy_X=False, positive=True)
    tomorrowModelLR.fit(XTomorrowLR, yTomorrowLR)
    # Predict tomorrow
    tomorrowPredictionLR = tomorrowModelLR.predict(XTomorrowLR_test)

    return tomorrowPredictionLR[0][0]


def modelBuildingSVR(originalAndPredData: pd.DataFrame, cleanedData: pd.DataFrame, split_date: str) -> float:
    '''Support Vector Regression Model'''
    # Making y and X for further usage in the model
    yTomorrowSVR = cleanedData.Pred.loc[split_date:]
    yTomorrowSVR = yTomorrowSVR.reset_index()
    yTomorrowSVR.drop(['Date'], axis=1, inplace=True)
    XTomorrowSVR = cleanedData.loc[split_date:]
    XTomorrowSVR.reset_index(inplace=True)
    XTomorrowSVR.drop(['Date', 'Pred'], axis=1, inplace=True)
    # Make today data as test data
    XTomorrowSVR_test = originalAndPredData.tail(1).reset_index()
    XTomorrowSVR_test.drop(['Date', 'Pred', 'stockSplits'], axis=1, inplace=True)
    # Making the model and fit it
    tomorrowModelSVR = SVR()
    tomorrowModelSVR.fit(XTomorrowSVR, yTomorrowSVR.values.ravel())
    # Predict tomorrow
    tomorrowPredictionSVR = tomorrowModelSVR.predict(XTomorrowSVR_test)

    return tomorrowPredictionSVR[0]





def tickerPrediction(ticker: str, timeFrame: str = 'D', todayPriceShow: bool = False) -> dict:
    '''
    Predict specific ticker with desired time frame and return a dictionary as prediction
    timeFrame -> str:
        Valid options:
            D: Daily
            W: Weekly
    '''
    out = {} # Store the data
    out['Ticker'] = ticker
    
    # Prepare the data
    originalAndPredDF, cleanedDF, t = dataPreparation(ticker, timeFrame)
    # Show today's price id necessary
    if todayPriceShow:
        print(f"\n{'*'*30}\n{'*'*30}\nToday Prices:\n{originalAndPredDF.iloc[-1, :5]}\n{'*'*30}\n{'*'*30}\n")
    
    # Predict
    out['RF'] = modelBuildingRF(originalAndPredDF, cleanedDF, t)
    out['Prophet_y'], out['Prophet_yUpper'], out['Prophet_yLower'] = modelBuildingProphet(originalAndPredDF, cleanedDF, t)
    out['LR'] = modelBuildingLR(originalAndPredDF, cleanedDF, t)
    out['SVR'] = modelBuildingSVR(originalAndPredDF, cleanedDF, t)

    return out


if __name__ == '__main__':
    # Get the data
    ticker = getTicker()
    tFrame = input('D: Daily\tW: Weekly\nPlease enter the desired time frame: ').upper()
    if tFrame not in ['D', 'W']:
        sys.exit('Invalid input for time frame.\nTry Again!!!')

    out = tickerPrediction(ticker, tFrame, True)

    # print(out)
    resultTXT = f"\n{'*'*30}\n{'*'*30}\n {out['Ticker']} Prediction results:\n\n\n\n Random Forest Prediction: {out['RF']}\n\n Linear Regression Prediction: {out['LR']}\n\n Support Vector Regression Prediction: {out['SVR']}\n\n Prophet Prediction:\n\tLower bound: {out['Prophet_yLower']}\n\tMain result: {out['Prophet_y']}\n\tUpper bound: {out['Prophet_yUpper']}"
    print(resultTXT)
