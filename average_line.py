from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib import ticker as mticker
#from matplotlib.finance import candlestick_ohlc
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, YEARLY
from matplotlib.dates import MonthLocator, MONTHLY
import datetime as dt
import pylab
from movingaverage import movingaverage
import quandl
#auth_token = open('auth_tok.txt', 'r').read()

MA1 = 10
MA2 = 50

# str(auth_token, encoding = "utf-8").replace('\n', '') #
quandl.ApiConfig.api_key = ""


def readstkData(stockcode, sday=None, eday=None):

    returndata = pd.DataFrame()

    returndata = quandl.get(stockcode)  # , authtoken=auth_token
    # Wash data
    # returndata = returndata.sort_index()
    # returndata.index.name = 'DateTime'
    # returndata.drop('VWAP', axis=1, inplace = True)
    # returndata.drop('Ask', axis=1, inplace = True)
    # returndata.columns = ['Bid', 'High', 'Last', 'Low', 'Volume']

    #returndata = returndata[returndata.index < eday.strftime('%Y-%m-%d')]

    return returndata


def main2():
    data = readstkData("BITSTAMP/USD")
    #data_log = np.log(data.Last.values)
    # rollmean10 = data_log.rolling(10).mean() #pd.rolling_mean(data['Last'], window=10)
    # rollmean5 = data_log.rolling(5).mean() #pd.rolling_mean(data['Last'], window=5)
    Av1 = list(movingaverage(data.Last.values, MA1))
    Av2 = list(movingaverage(data.Last.values, MA2))
    #Av2 = movingaverage(data.Last.values, MA2)
    #rollstd = pd.rolling_std(data['Last'], window=12)
    plt.plot(data['Last'], color='blue', label='Original')
    plt.plot(data['Last'].index[MA1-1:], Av1,
             color='red', label='Rolling Mean 10d')
    plt.plot(data['Last'].index[MA2-1:], Av2,
             color='green', label='Rolling Mean 50d')
    #plt.plot(Av1, color='green', label='Rolling Mean 5d')
    #plt.plot(rollstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')

    # plt.plot(data['Last'])
    plt.title('Plot price & mean')
    plt.show()


if __name__ == "__main__":
    main2()
