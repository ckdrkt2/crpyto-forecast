import pymysql
import pandas as pd
import yfinance as yf
import datetime
import time

# database
conn = pymysql.connect(host='192.168.153.110', port=31802, user='root', password='tmaxfintech', db='finance', charset='utf8', autocommit=True, cursorclass=pymysql.cursors.DictCursor)
cur = conn.cursor()

today = datetime.date.today()
stamp = int(time.mktime(datetime.date.today().timetuple()))

info = {'ADA': 7, 'BOND10Y': 6, 'BOND1M': 6, 'BOND1Y': 6, 'BOND30Y': 6, 'BOND5Y': 6, 'BOND6M': 6, 'BTC': 7, 'DOW': 7, 'ETH': 7, 'EU': 7, 'FRANCE': 7, 'GAS': 7, 'GERMANY': 6, 'GOLD': 7, 'HG': 7, 'KOSPI': 7, 'NIKKEI': 6, 'OIL': 7, 'SP500': 6, 'SSEC': 7, 'UK': 6, 'USD/EUR': 6, 'USD/KRW': 6, 'USD/RUB': 6, 'VIX': 6, 'XRP': 7}
symbols = {'ADA': 'ADA-USD', 'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'XRP': 'XRP-USD', 'BOND5Y': '^FVX', 'BOND10Y': '^TNX', 'BOND30Y': '^TYX', 'DOW': '^DJI', 'NG': 'NG=F', 'GOLD': 'GC=F', 'KOSPI': '^KS11', 'NIKKEI': '^N225', 'OIL': 'CL=F', 'SP500': '^GSPC', 'USD/EUR': 'EUR=X', 'USD/RUB': 'RUB=X', 'USD/KRW': 'KRW=X', 'VIX': '^VIX', 'UK': '^FTSE', 'GERMANY': '^GDAXI', 'HG': 'QC=F'}

for col in info:
    if col not in symbols: continue

    symbol = symbols[col]
    df = yf.download(symbol, start=today)
    df = df.reset_index()
    data = dict(df.iloc[0])

    if info[col] == 7:
        query = "insert into `{}`(timestamp, Date, Close, Open, High, Low, volume) values (%s, %s, %s, %s, %s, %s, %s)".format(col)
        cur.execute(query, [stamp, today, data['Close'], data['Open'], data['High'], data['Low'], data['Volume']])

    else:
        query = "insert into `{}`(timestamp, Date, Close, Open, High, Low) values (%s, %s, %s, %s, %s, %s)".format(col)
        cur.execute(query, [stamp, today, data['Close'], data['Open'], data['High'], data['Low']])

    cur.fetchall()