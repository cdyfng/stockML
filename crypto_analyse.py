import os
import numpy as np
import pandas as pd
import pickle
import quandl
from datetime import datetime
import talib as ta
from sklearn import svm, preprocessing, linear_model
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble


def get_quandl_data(quandl_id):
    '''Download and cache Quandl dataseries'''
    cache_path = '{}.pkl'.format(quandl_id).replace('/', '-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)
        print('Loaded {} from cache'.format(quandl_id))
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df


def get_json_data(json_url, cache_path):
    '''Download and cache JSON data, return as a dataframe.'''
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)
        print('Loaded {} from cache'.format(json_url))
    except (OSError, IOError) as e:
        print('Downloading {}'.format(json_url))
        df = pd.read_json(json_url)
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(json_url, cache_path))
    return df


base_polo_url = 'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}'
# get data from the start of 2015
start_date = datetime.strptime('2015-01-01', '%Y-%m-%d')
end_date = datetime.now()  # up until today
pediod = 86400  # pull daily data (86,400 seconds per day)


def get_crypto_data(poloniex_pair):
    '''Retrieve cryptocurrency data from poloniex'''
    json_url = base_polo_url.format(
        poloniex_pair, start_date.timestamp(), end_date.timestamp(), pediod)
    data_df = get_json_data(json_url, poloniex_pair)
    data_df = data_df.set_index('date')
    return data_df

  #btc_usd_price_binance = get_quandl_data('BCHARTS/KRAKENUSD')


def merge_dfs_on_column(dataframes, labels, col):
    '''Merge a single column of each dataframe into a new combined dataframe'''
    series_dict = {}
    for index in range(len(dataframes)):
        series_dict[labels[index]] = dataframes[index][col]

    return pd.DataFrame(series_dict)


# altcoins = ['ETH','LTC','XRP','ETC','STR','DASH','SC','XMR','XEM']

# altcoin_data = {}
# for altcoin in altcoins:
#     coinpair = 'BTC_{}'.format(altcoin)
#     crypto_price_df = get_crypto_data(coinpair)
#     altcoin_data[altcoin] = crypto_price_df


# for altcoin in altcoin_data.keys():
#     altcoin_data[altcoin]['price_usd'] =  altcoin_data[altcoin]['weightedAverage'] * btc_usd_datasets['avg_btc_price_usd']


# one_year_combined_df = combined_df[combined_df.index > '2017-08-01']

# pyplot.yscale('log')
#one_year_combined_df['BTC'] = btc_usd_price_binance['Weighted Price']

# for i in range(len(test_df)-1):
# 	test_df.iloc[i+1]['Nplut1Weighted Price'] = test_df.iloc[i+1]['Weighted Price']

# 	test_df.loc['2018-07-30', 'Nplut1Weighted Price'] = 123


btc_usd_price_binance = get_quandl_data('BCHARTS/KRAKENUSD')
tmpWeight = btc_usd_price_binance['Weighted Price'].shift(-1)
btc_usd_price_binance['OneDayGain'] = tmpWeight / \
    btc_usd_price_binance['Weighted Price']


def generate_index():
        #cci30 = pd.read_csv('./cci30_OHLCV.csv')
    cci30 = pd.read_csv('https://cci30.com/ajax/getIndexHistory.php')
    cci30['Date'] = pd.to_datetime(cci30['Date'])
    cci30.set_index('Date', inplace=True)
    tmpClose = cci30['Close'].shift(1)
    cci30['DailyGain'] = tmpClose/cci30['Close']
    return cci30


cci30 = generate_index()

btc_usd_price_binance['cci30DaylyGain'] = cci30.DailyGain
btc_usd_price_binance = btc_usd_price_binance[btc_usd_price_binance.cci30DaylyGain > 0]
btc_usd_price_binance['Difference'] = (
    btc_usd_price_binance.OneDayGain - btc_usd_price_binance.cci30DaylyGain > 0.02).replace(True, 1).replace(False, 0)
btc_usd_price_binance['GainDiff'] = btc_usd_price_binance.OneDayGain - \
    btc_usd_price_binance.cci30DaylyGain


rsi = ta.RSI(np.array(btc_usd_price_binance.Close), 14)
sma = ta.SMA(np.array(btc_usd_price_binance.Close))

var = ta.VAR(np.array(btc_usd_price_binance.Close))
btc_usd_price_binance['Var'] = var
btc_usd_price_binance['Rsi14'] = rsi
btc_usd_price_binance['Sma'] = sma


def toStatCSV():
    pass
    # process the btc and altcoin data
    # generate a csv with formate:  index price date dailyRate coinName


data_df = btc_usd_price_binance
data_df = data_df.replace('N/A', 0).replace(np.nan, 0).replace(np.inf, 0)


# 线性回归
linear_reg = linear_model.LinearRegression()
# 树回归
tree_reg = tree.DecisionTreeRegressor()
# SVM回归
svr = svm.SVR()
# KNN
knn = neighbors.KNeighborsRegressor()
# 随机森林
rf = ensemble.RandomForestRegressor(n_estimators=20)
# Adaboost
ada = ensemble.AdaBoostRegressor(n_estimators=50)
# GBRT
gbrt = ensemble.GradientBoostingRegressor(n_estimators=100)
# 大部分学习方法不是针对0，1分类， 那么如果不用0，1来作为y来预测，就直接用盈利或亏损的百分比来做预测，来测试一下所有的方法。


def Mlearn(data_df):

    #features =  ['Weighted Price',]
    features = ['Rsi14', 'Var', 'Sma', 'Volume (BTC)', 'Volume (Currency)']

    X = data_df[features]
    #X = preprocessing.scale(X).values

    X = preprocessing.scale(X)
    #X = data_df[features].values
    y = data_df.Difference.values
    y = data_df.GainDiff.values

    #clf = svm.SVC(kernel='linear', C=1.0)
    test_size = 100  # linear 9000 相当于800多数据学习，耗时1分钟左右 # 9800 4纬可运行
    #clf = svm.SVC(kernel='rbf', tol=1e-5, C=1.0)
    #clf = svm.SVC(kernel='linear', C=1.0)
    #clf = svm.SVC(gamma=0.001)
    # 使用 linear结果学习过的数据 预测还是0； rbf学习过的数据是对的，其他全为0
    clf = knn
    clf.fit(X[:-test_size], y[:-test_size])

    pred = clf.predict(X[-test_size:])
    check_y = y[-test_size:]
    correct_count = 0
    for x in range(test_size):
        if pred[x] == y[x]:
            correct_count += 1
    print(correct_count/test_size)

    correct_count = 0
    for x in range(1200):
        if pred[x] == y[x]:
            correct_count += 1


# def plot():
# result = pred
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(np.arange(len(result)), pred,'go-',label='true value')
# plt.plot(np.arange(len(result)), check_y, 'ro-',label='predict value')
# plt.legend()
# plt.show()


# check_direction = check_y/pred

# >>> len(check_direction[check_direction > 0])
# 48
# >>> len(check_direction[check_direction < 0])
# 51


def try_different_method(clf, plt, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    result = clf.predict(x_test)
    plt.figure()
    print(result)
    check_direction = result/y_test
    print(len(check_direction[check_direction > 0]),
          len(check_direction[check_direction < 0]))
    plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
    plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
    plt.title('score: %f' % score)
    plt.legend()
    plt.show()


#try_different_method(rf, plt, X[:-test_size], y[:-test_size], X[-test_size:], y[-test_size:])

#try_different_method(ada, plt, X[:-test_size], y[:-test_size], X[-test_size:], y[-test_size:])

features = ['Rsi14', 'Var', 'Sma', 'Volume (BTC)', 'Volume (Currency)']
X = data_df[features]
X = preprocessing.scale(X)
#X = data_df[features].values
#y = data_df.Difference.values
y = data_df.GainDiff.values
test_size = 100

try_different_method(knn, plt, X[:-test_size],
                     y[:-test_size], X[-test_size:], y[-test_size:])
