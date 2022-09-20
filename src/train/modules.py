import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import pickle

def get_features(df):
    df['Target'] = df['Close'].shift(-360)
    df = df.dropna()
    return df

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

def plot_importance(importances, features_names, PLOT_TOP_N=20, figsize=(10, 10)):
    importance_df = pd.DataFrame(data=importances, columns=features_names)
    sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    ax.set_xscale('log')
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance')
    sns.boxplot(data=sorted_importance_df[plot_cols], orient='h', ax=ax)
    plt.savefig("feature_importance.png", dpi=120)
    plt.show()

def get_time_series_cross_val_splits(data, cv=7, embargo=3750):
    all_train_timestamps = data['Time'].unique()
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


def get_Xy_and_model_for_asset(df_proc, model_type, n_fold, features, params):
    # EmbargoCV
    train_test_zip = get_time_series_cross_val_splits(df_proc, cv=n_fold, embargo=3750)
    print("entering time series cross validation loop")
    importances = []

    for split, train_test_split in enumerate(train_test_zip):
        print(f"doing split {split + 1} out of {n_fold}")
        train_split, test_split = train_test_split
        train_split_index = df_proc['Time'].isin(train_split)
        test_split_index = df_proc['Time'].isin(test_split)

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
                          verbose_eval=100,
                          feval=correlation,
                          )
        importances.append(model.feature_importance(importance_type='gain'))

        file = f'trained_model_fold{split}.pkl'
        pickle.dump(model, open(file, 'wb'))
        print(f"Trained model was saved to 'trained_model_fold{split}.pkl'")
        print("")

    plot_importance(np.array(importances), features, PLOT_TOP_N=20, figsize=(10, 5))