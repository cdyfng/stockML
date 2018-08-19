#from Quandl import Quandl
import quandl
import pandas as pd
import os
import time
from datetime import date, timedelta, datetime

path = '/Users/fong/work/quant/stockML/data/'
#auth_token = open('auth_tok.txt', 'r').read()

def Stock_Prices():
    df = pd.DataFrame()
    statspath = path+'Yahoo/intraQuarter/_KeyStats'
    stock_list = [x[0] for x in os.walk(statspath)]
    print('total', len(stock_list))
    stock_list = stock_list[-100:]

    for each_dir in stock_list[1:]:
        try:

            ticker = each_dir.split('_KeyStats/')[1]
            print(ticker)
            name = 'WIKI/' + ticker.upper()
            data = quandl.get(name, trim_start=datetime.strptime('2000-1-1', '%Y-%m-%d'),
                          trim_end=date.today()-timedelta(1),)
                          # authtoken=auth_token
            data[ticker.upper()] = data['Adj. Close']

            df = pd.concat([df, data[ticker.upper()]], axis=1)

        except Exception as e:
            print('Error polling Quandl: ' +str(e) + each_dir)

    df.to_csv('./data/Quandl/stock_prices.csv')



Stock_Prices()

