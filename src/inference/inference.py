import pandas as pd
import pickle
import pymysql
import json
import datetime

n_fold = 7
symbol = "ADAUSDT"
interval = "1m"

# load data
conn = pymysql.connect(host='192.168.153.110', port=31802, user='root', password='tmaxfintech', db='COINS', charset='utf8', autocommit=True, cursorclass=pymysql.cursors.DictCursor)
cur = conn.cursor()
cur.execute("select max(Time) from ADA;")
target_time = cur.fetchall()[0]['max(Time)']
cur.execute("select * from ADA where Time = %s;", (target_time))
df = pd.DataFrame(cur.fetchall()).drop(['Time'], axis=1)

# model result
models = []
for i in range(n_fold):
    with open('models/trained_model_fold{}.pkl'.format(i), 'rb') as f:
        model = pickle.load(f)
    models.append(model.predict(df))

avg_of_model = sum(models) / n_fold
date = str(datetime.datetime.fromtimestamp(target_time // 1000))
print(date, ":", avg_of_model[0])