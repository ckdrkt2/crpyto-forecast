import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import time
import datetime
import json
import pickle
import pymysql
from sklearn.metrics import mean_squared_error
from modules import get_features, get_time_series_cross_val_splits, get_Xy_and_model_for_asset, correlation, corr_score, wcorr_score, plot_importance

n_fold = 7

params = {
    'goss': {
        'num_boost_round': 5000,
        'early_stopping_rounds': 100,
        'objective': 'regression_l2',
        'metric': 'rmse',
        'boosting_type': 'goss',
        'max_depth': -1,
        'num_leaves': 127,
        'max_bin': 600,
        'min_data_in_leaf': 50,
        'learning_rate': 0.003,
        'feature_fraction': 1,
        'lambda_l1': 0.5,
        'lambda_l2': 2,
        "verbose": -1,
        "first_metric_only": True,
    },
}

# load data
conn = pymysql.connect(host='192.168.153.110', port=31802, user='root', password='tmaxfintech', db='COINS', charset='utf8', autocommit=True, cursorclass=pymysql.cursors.DictCursor)
cur = conn.cursor()
cur.execute("select * from ADA where Time >= 1656601200000;")
df = pd.DataFrame(cur.fetchall())

# feature engineering
feat = get_features(df)
not_use_features_train = ['Time', 'Target']
features = feat.columns
features = features.drop(not_use_features_train)
features = list(features)

# train
train = feat
get_Xy_and_model_for_asset(train, 'goss', n_fold, features, params)

# model result
models = []
for i in range(n_fold):
    with open('trained_model_fold{}.pkl'.format(i), 'rb') as f:
        model = pickle.load(f)
    models.append(model.predict(train))

avg_of_model = sum(models) / n_fold
model_df = pd.concat([pd.DataFrame(avg_of_model), train['Target']], axis=1)
model_df.columns = ['predict', 'target']
rmse, corr = mean_squared_error(train['Target'], avg_of_model) ** 0.5, model_df.corr()['predict']['target']
print("RMSE: ", rmse, 'corr: ', corr)
with open("score.json", "w") as outfile:
    json.dump({'RMSE': rmse, "corr": corr}, outfile)
plt.savefig("result.png", dpi=120)
# model_df.plot()
