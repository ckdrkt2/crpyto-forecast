import pymysql
import requests
import pandas as pd
from sqlalchemy import create_engine

# database
conn = pymysql.connect(host='192.168.153.110', port=31802, user='root', password='tmaxfintech', db='COINS', charset='utf8', autocommit=True, cursorclass=pymysql.cursors.DictCursor)
cur = conn.cursor()

cur.execute("select max(Time) from ADA;")

last = cur.fetchall()[0]['max(Time)'] + 60000
symbol = 'ADAUSDT'
interval = '1m'

url = "https://api.binance.com/api/v3/uiKlines?symbol={}&interval={}&startTime={}&limit=1000".format(symbol, interval, last)
headers = {"accept": "application/json"}
response = requests.get(url, headers=headers)

df = pd.DataFrame(response.json())
df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote', 'Trades', 'tb_base_av', 'tb_quote_av', 'Unused']
df = df.drop(['Unused', 'Close Time'], axis=1)

db_connection = create_engine('mysql+pymysql://root:tmaxfintech@192.168.153.110:31802/COINS')
df.to_sql(name='ADA', con=db_connection, if_exists='append', index=False)

# symbol = 'ADAUSDT'
# interval = '1m'
#
# stamp = int(time.mktime(target.timetuple())*1000)

