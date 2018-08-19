import os
import urllib3
import time

path = '/Users/fong/work/quant/stockML/data'

def Check_Yahoo():
    statspath = path + '/Yahoo/intraQuarter/_KeyStats/'
    stock_list = [x[0] for x in os.walk(statspath)]

    try:
        #print(stock_list)
        #stock_list = stock_list[-3:]
        http = urllib3.PoolManager()
        for e in stock_list[1:]:
            e = e.replace(statspath, '')
            #print(e, '\n')
            if e < 'znga':
                link = 'http://finance.yahoo.com/q/ks?s=%s+Key+Statistics' % e.upper()
                #response = urllib3.urlopen(link)
                #html = response.read()
                r = http.request('GET', link)
                html = r.data
                print(e, link)
                save = '%s/Yahoo/forward/%s.html' % (path, str(e))
                store = open(save, 'w')
                store.write(str(html))
                store.close()
                time.sleep(1)

    except Exception as e:
        print(str(e))

Check_Yahoo()

#raw_input('Enter to exit')


