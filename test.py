import os
import random
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from FinanceDataReader import DataReader
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import lightgbm as lgb
import pickle
from sklearn.metrics import mean_squared_error

n_fold = 10

params = {
    'num_boost_round': 5000,
    'early_stopping_rounds': 500,
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'goss',
    'max_depth': 7,
    'num_leaves': 127,
    'max_bin': 600,
    'min_data_in_leaf': 50,
    'learning_rate': 0.03,
    'feature_fraction': 1,
    'lambda_l1': 0.5,
    'lambda_l2': 2,
    "verbose": -1,
    "first_metric_only": True,
}

target_date = '2018-02-20'

data = pd.read_csv('data.csv')
data = data.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1)
data['target'] = np.log10(data['Close'].shift(-1))
data = data.dropna()

data['timestamp'] = [int(i.timestamp()) for i in pd.date_range(start=target_date, end='2022-09-14').tolist()]

feat = data
not_use_features_train = ['target', 'timestamp', 'Date', 'Open', 'High', 'Low', 'Close']
features = feat.columns
features = features.drop(not_use_features_train)
features = list(features)

def correlation(a, train_data):
    b = train_data.get_label()

    a = np.ravel(a)
    b = np.ravel(b)

    len_data = len(a)
    mean_a = np.sum(a) / len_data
    mean_b = np.sum(b) / len_data
    var_a = np.sum(np.square(a - mean_a)) / len_data
    var_b = np.sum(np.square(b - mean_b)) / len_data

    cov = np.sum((a * b)) / len_data - mean_a * mean_b
    corr = cov / np.sqrt(var_a * var_b)

    return 'corr', corr, True

def corr_score(pred, valid):
    len_data = len(pred)
    mean_pred = np.sum(pred) / len_data
    mean_valid = np.sum(valid) / len_data
    var_pred = np.sum(np.square(pred - mean_pred)) / len_data
    var_valid = np.sum(np.square(valid - mean_valid)) / len_data

    cov = np.sum((pred * valid)) / len_data - mean_pred * mean_valid
    corr = cov / np.sqrt(var_pred * var_valid)

    return corr

def wcorr_score(pred, valid, weight):
    len_data = len(pred)
    sum_w = np.sum(weight)
    mean_pred = np.sum(pred * weight) / sum_w
    mean_valid = np.sum(valid * weight) / sum_w
    var_pred = np.sum(weight * np.square(pred - mean_pred)) / sum_w
    var_valid = np.sum(weight * np.square(valid - mean_valid)) / sum_w

    cov = np.sum((pred * valid * weight)) / sum_w - mean_pred * mean_valid
    corr = cov / np.sqrt(var_pred * var_valid)

    return corr


def plot_importance(importances, features_names=features, PLOT_TOP_N=20, figsize=(10,10)):
    importance_df = pd.DataFrame(data=importances, columns=features)
    sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    ax.set_xscale('log')
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance')
    sns.boxplot(data=sorted_importance_df[plot_cols],
                orient='h',
                ax=ax)
    plt.show()
    plt.savefig('feature_importance.png')

def get_time_series_cross_val_splits(data, cv=n_fold, embargo=3):
    all_train_timestamps = data['timestamp'].unique()
    len_split = len(all_train_timestamps) // cv
    test_splits = [all_train_timestamps[i * len_split:(i + 1) * len_split] for i in range(cv)]
    rem = len(all_train_timestamps) - len_split*cv
    if rem>0:
        test_splits[-1] = np.append(test_splits[-1], all_train_timestamps[-rem:])

    train_splits = []
    for test_split in test_splits:
        test_split_max = int(np.max(test_split))
        test_split_min = int(np.min(test_split))
        train_split_not_embargoed = [e for e in all_train_timestamps if not (test_split_min <= int(e) <= test_split_max)]
        embargo_sec = 60*embargo
        train_split = [e for e in train_split_not_embargoed if
                       abs(int(e) - test_split_max) > embargo_sec and abs(int(e) - test_split_min) > embargo_sec]
        train_splits.append(train_split)

    train_test_zip = zip(train_splits, test_splits)
    return train_test_zip


def get_Xy_and_model_for_asset(df_proc):
    # EmbargoCV
    train_test_zip = get_time_series_cross_val_splits(df_proc, cv=n_fold, embargo=30)
    print("entering time series cross validation loop")
    importances = []

    for split, train_test_split in enumerate(train_test_zip):
        print(f"doing split {split + 1} out of {n_fold}")
        train_split, test_split = train_test_split
        train_split_index = df_proc['timestamp'].isin(train_split)
        test_split_index = df_proc['timestamp'].isin(test_split)

        train_dataset = lgb.Dataset(df_proc.loc[train_split_index, features],
                                    df_proc.loc[train_split_index, f'target'].values,
                                    feature_name=features,
                                    )
        val_dataset = lgb.Dataset(df_proc.loc[test_split_index, features],
                                  df_proc.loc[test_split_index, f'target'].values,
                                  feature_name=features,
                                  )

        print(f"number of train data: {len(df_proc.loc[train_split_index])}")
        print(f"number of val data:   {len(df_proc.loc[test_split_index])}")

        model = lgb.train(params=params,
                          train_set=train_dataset,
                          valid_sets=[train_dataset, val_dataset],
                          valid_names=['tr', 'vl'],
                          verbose_eval = 100,
                          feval=correlation,
                          )
        importances.append(model.feature_importance(importance_type='gain'))

        file = f'trained_model_fold{split}.pkl'
        pickle.dump(model, open(file, 'wb'))
        print(f"Trained model was saved to 'trained_model_fold{split}.pkl'")
        print("")

    plot_importance(np.array(importances), features, PLOT_TOP_N=30, figsize=(10, 10))
target_stamp = int(datetime.datetime(2022,1,1).timestamp())
train, test = feat[feat['timestamp'] < target_stamp], feat[feat['timestamp'] >= target_stamp]
get_Xy_and_model_for_asset(train)

# ensemble fold models
models = []

testX = test.drop(not_use_features_train, axis=1)
testY = test['target'].reset_index(drop=True)

good = 0
for i in range(n_fold):
    with open('./trained_model_fold{}.pkl'.format(i), 'rb') as f:
        model = pickle.load(f)
    result = model.predict(testX, predict_disable_shape_check=True)
    model_df = pd.concat([pd.DataFrame(result), testY], axis=1)
    model_df.columns = ['predict', 'target']
    rmse = mean_squared_error(testY, pd.DataFrame(result)) ** 0.5
    print("RMSE: ", rmse, 'corr: ', model_df.corr()['predict']['target'])

    if rmse < 0.3:
        models.append(result)
        good += 1

avg_of_model = sum(models) / good
model_df = pd.concat([pd.DataFrame(avg_of_model), testY], axis=1)
model_df.columns = ['predict', 'target']
print("good models -> RMSE: ", mean_squared_error(testY, avg_of_model) ** 0.5, 'corr: ', model_df.corr()['predict']['target'])
model_df.plot()
plt.savefig('plot.png')