# stockTickerPrediction

Predict tomorrow’s price of specific company in US stock market using different algorithms.

One can find this repo helpful if you want to make familiar with different Machine Learning algorithms or how to get along with finance using python.

## Technologies

---

**Python 3.8**
#### Required libraries
- numpy
- pandas
- matplotlib
- seaborn
- plotly
- IPython
- sklearn
- fbprophet
- yfinance
- built-in libs:
  - sys

## How to use

---

- indicators
  * You can find several technical indicators in this module.
- featureCreator
  * Make new features using our main data (OHLC + Volume)
- Ticker EDA
  * See this notebook to get the insights of how to predict stock prices.
  * This notebook explore "GOOGL" data.
  * Also you can show your desired company's ticker data. (Change cell 2)
- tickerPrediction
  * You can find the simplified version of whole process in Ticker EDA in this module.

## Acknowledgments

---

* These scripts and notebooks were built using [yahoo!finance](https://finance.yahoo.com/) data that fetch from [yfinance](https://github.com/ranaroussi/yfinance) package.
* Thanks to [Mike](https://github.com/mtodisco10) who helped me with this project.