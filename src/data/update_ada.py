import pymysql
import requests
import pandas as pd
from sqlalchemy import create_engine
from time import sleep

# get last data timestamp
conn = pymysql.connect(host='192.168.153.110', port=31802, user='root', password='tmaxfintech', db='COINS', charset='utf8', autocommit=True, cursorclass=pymysql.cursors.DictCursor)
cur = conn.cursor()
cur.execute("select max(Time) from ADA;")
start = cur.fetchall()[0]['max(Time)'] + 60000

# update ada data
symbol = 'ADAUSDT'
interval = '1m'

while True:

    url = "https://api.binance.com/api/v3/uiKlines?symbol={}&interval={}&startTime={}&limit=1000".format(symbol, interval, start)
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)

    df = pd.DataFrame(response.json())

    if len(df) > 0:
        df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote', 'Trades', 'tb_base_av', 'tb_quote_av', 'Unused']
        df = df.drop(['Unused', 'Close Time'], axis=1)
        df = df.astype({'Open': 'float', 'High': 'float', 'Low': 'float', 'Close': 'float', 'Volume': 'float', 'Quote': 'float', 'tb_quote_av': 'float', 'tb_base_av': 'float'})

        db_connection = create_engine('mysql+pymysql://root:tmaxfintech@192.168.153.110:31802/COINS')
        df.to_sql(name='ADA', con=db_connection, if_exists='append', index=False)

    sleep(20)