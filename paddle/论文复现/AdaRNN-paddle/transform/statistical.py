import datetime

import numpy as np
import pandas as pd


def get_data_statistic(df, station, start_time, end_time):
    mean_train, std_train = get_dataset_statistic(
        df, station, start_time, end_time)
    return mean_train, std_train


def get_dataset_statistic(df, station, start_date, end_date):
    data= df[station]
    feat, label = data[0], data[1]
    referece_start_time = datetime.datetime(2013, 3, 1, 0, 0)
    referece_end_time = datetime.datetime(2017, 2, 28, 0, 0)

    assert (pd.to_datetime(start_date) - referece_start_time).days >= 0
    assert (pd.to_datetime(end_date) - referece_end_time).days <= 0
    assert (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days >= 0
    index_start = (pd.to_datetime(start_date) - referece_start_time).days
    index_end = (pd.to_datetime(end_date) - referece_start_time).days
    feat = feat[index_start: index_end + 1]
    # label = label[index_start: index_end + 1]
    feat = feat.reshape(-1, feat.shape[2])
    mu_train = np.mean(feat, axis=0)
    sigma_train = np.std(feat, axis=0)

    return mu_train, sigma_train


def get_data_statistic_welldata(df):
    data = df
    feat, label = data[0], data[1]
    referece_start_time = datetime.datetime(2013, 3, 1, 0, 0)
    referece_end_time = datetime.datetime(2017, 2, 28, 0, 0)

    assert (pd.to_datetime(start_date) - referece_start_time).days >= 0
    assert (pd.to_datetime(end_date) - referece_end_time).days <= 0
    assert (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days >= 0
    index_start = (pd.to_datetime(start_date) - referece_start_time).days
    index_end = (pd.to_datetime(end_date) - referece_start_time).days
    feat = feat[index_start: index_end + 1]
    # label = label[index_start: index_end + 1]
    feat = feat.reshape(-1, feat.shape[2])
    mu_train = np.mean(feat, axis=0)
    sigma_train = np.std(feat, axis=0)

    return mu_train, sigma_train