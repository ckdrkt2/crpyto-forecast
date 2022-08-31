import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import time
import datetime
import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

n_fold = 7
seed0 = 8586

params = {
    'goss' : {
        'num_boost_round': 100,
        'early_stopping_rounds': 100,
        'objective': 'regression_l2',
        'metric': 'rmse',
        'boosting_type': 'goss',
        'max_depth': -1,
    #     'num_leaves': 127,
        'verbose': -1,
        'max_bin': 255,
        'min_data_in_leaf':1000,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'lambda_l1': 0,
        'lambda_l2': 2,
        'seed':seed0,
        'feature_fraction_seed': seed0,
        'bagging_fraction_seed': seed0,
        'drop_seed': seed0,
        'data_random_seed': seed0,
        'extra_trees': True,
        'extra_seed': seed0,
        'zero_as_missing': True,
        "first_metric_only": True,
        "device": "gpu",
    },
    'gbdt' : {
        'num_boost_round': 1000,
        'early_stopping_rounds': 50,
    #     'objective': 'regression',
        'objective': 'regression_l2',
        'metric': 'rmse',
    #     'boosting_type': 'gbdt',
        'boosting_type': 'goss',
        'max_depth': -1,
    #     'num_leaves': 127,
        'verbose': -1,
        'max_bin': 255,
        'min_data_in_leaf':1000,
        'learning_rate': 0.03,
    #     'subsample': 0.8,
    #     'subsample_freq': 1,
        'feature_fraction': 0.8,
        'lambda_l1': 0,
        'lambda_l2': 2,
        'seed':seed0,
        'feature_fraction_seed': seed0,
        'bagging_fraction_seed': seed0,
        'drop_seed': seed0,
        'data_random_seed': seed0,
        'extra_trees': True,
        'extra_seed': seed0,
        'zero_as_missing': True,
        "first_metric_only": False,
        "device": "gpu",
    }
}

df_train = pd.DataFrame()
for year in range(2022, 2023):
    data0 = pd.read_csv('data/ada_{}.csv'.format(year))
    df_train = pd.concat([df_train, data0])
df_close = df_train[['Open_time', 'Close']].shift(-360)
df_train['Target'] = df_close['Close']
df_train = df_train.drop('Symbol', axis=1)
df_train = df_train.dropna()
df_train['timestamp'] = pd.to_datetime(df_train['Open_time'], format='%Y-%m-%d %H:%M:%S').astype('int64') // 1000000000


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


feat = df_train
feat = feat.drop(['Open', 'High', 'Low', 'Volume', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av'], axis=1)
# feat = reduce_mem_usage(feat)

not_use_features_train = ['Open_time', 'Target', 'timestamp']
features = feat.columns
features = features.drop(not_use_features_train)
features = list(features)


# define the evaluation metric
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


# For CV score calculation
def corr_score(pred, valid):
    len_data = len(pred)
    mean_pred = np.sum(pred) / len_data
    mean_valid = np.sum(valid) / len_data
    var_pred = np.sum(np.square(pred - mean_pred)) / len_data
    var_valid = np.sum(np.square(valid - mean_valid)) / len_data

    cov = np.sum((pred * valid)) / len_data - mean_pred * mean_valid
    corr = cov / np.sqrt(var_pred * var_valid)

    return corr


# For CV score calculation
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


def plot_importance(importances, features_names=features, PLOT_TOP_N=20, figsize=(10, 10)):
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
    plt.savefig("feature_importance.png", dpi=120)
    # plt.show()

def get_time_series_cross_val_splits(data, cv=n_fold, embargo=3750):
    all_train_timestamps = data['timestamp'].unique()
    len_split = len(all_train_timestamps) // (cv + 1)
    splits = [all_train_timestamps[i * len_split:(i + 1) * len_split] for i in range(cv + 1)]

    rem = len(all_train_timestamps) - len_split * (cv + 1)
    if rem > 0: splits[-1] = np.append(splits[-1], all_train_timestamps[-rem:])

    train_splits = [splits.pop(0)]
    test_splits = []
    for split in splits:
        train_splits.append(np.concatenate([train_splits[-1], split]))
        test_splits.append(split)

    train_test_zip = zip(train_splits, test_splits)
    return train_test_zip


def get_Xy_and_model_for_asset(df_proc, model_type):
    # EmbargoCV
    train_test_zip = get_time_series_cross_val_splits(df_proc, cv=n_fold, embargo=3750)
    print("entering time series cross validation loop")
    importances = []

    for split, train_test_split in enumerate(train_test_zip):
        print(f"doing split {split + 1} out of {n_fold}")
        train_split, test_split = train_test_split
        train_split_index = df_proc['timestamp'].isin(train_split)
        test_split_index = df_proc['timestamp'].isin(test_split)

        train_dataset = lgb.Dataset(df_proc.loc[train_split_index, features],
                                    df_proc.loc[train_split_index, f'Target'].values,
                                    feature_name=features,
                                    )
        val_dataset = lgb.Dataset(df_proc.loc[test_split_index, features],
                                  df_proc.loc[test_split_index, f'Target'].values,
                                  feature_name=features,
                                  )

        print(f"number of train data: {len(df_proc.loc[train_split_index])}")
        print(f"number of val data:   {len(df_proc.loc[test_split_index])}")

        model = lgb.train(params=params[model_type],
                          train_set=train_dataset,
                          valid_sets=[train_dataset, val_dataset],
                          valid_names=['tr', 'vl'],
                          verbose_eval=10,
                          feval=correlation,
                          )
        importances.append(model.feature_importance(importance_type='gain'))

        file = f'trained_model_fold{split}.pkl'
        pickle.dump(model, open(file, 'wb'))
        print(f"Trained model was saved to 'trained_model_fold{split}.pkl'")
        print("")

    plot_importance(np.array(importances), features, PLOT_TOP_N=20, figsize=(10, 5))

get_Xy_and_model_for_asset(feat[feat['Open_time'] < '2022-07-01 00:00:00'], 'goss')

# ensemble fold models
models = []
target_date = '2022-07-01 00:00:00'
testX = feat[feat['Open_time'] >= target_date].drop(not_use_features_train, axis=1)
testY = feat[feat['Open_time'] >= target_date]['Target']
for i in range(7):
    with open('./trained_model_fold{}.pkl'.format(i), 'rb') as f:
        model = pickle.load(f)
    models.append(model.predict(testX))

avg_of_model = sum(models) / 7
model_df = pd.concat([pd.DataFrame(avg_of_model), testY], axis=1)
model_df.columns = ['predict', 'target']
print("RMSE: ", mean_squared_error(testY, avg_of_model)**0.5, 'corr: ', model_df.corr()['predict']['target'])
# model_df.plot()
