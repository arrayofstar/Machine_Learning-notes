# -*- coding: utf-8 -*-
# @Time    : 2022/10/23 21:10
# @Author  : Dreamstar
# @File    : dataloader_welldata.py
# @Desc    : 重写测井数据的dataloader


# -*- coding: utf-8 -*-
# @Time    : 2022/10/22 17:06
# @Author  : Dreamstar
# @File    : dataloader.py
# @Desc    :
import os

import pandas as pd
import torch

from transform import statistical
import datetime

from analysis.domain_analysis import tdc
from torch.utils.data import Dataset, DataLoader


class data_loader(Dataset):
    def __init__(self, df_feature, df_label, df_label_reg, transform=None):
        # 判断特征、标签、回归标签长度一致
        assert len(df_feature) == len(df_label)
        assert len(df_feature) == len(df_label_reg)

        self.df_feature = df_feature
        self.df_label = df_label
        self.df_label_reg = df_label_reg

        self.T = transform
        self.df_feature = torch.tensor(
            self.df_feature, dtype=torch.float32)
        self.df_label = torch.tensor(
            self.df_label, dtype=torch.float32)
        self.df_label_reg = torch.tensor(
            self.df_label_reg, dtype=torch.float32)

    def __getitem__(self, index):
        sample, target, label_reg = self.df_feature[index], self.df_label[index], self.df_label_reg[index]
        if self.T:
            return self.T(sample), target
        else:
            return sample, target, label_reg

    def __len__(self):
        return len(self.df_feature)


def create_dataset(df, station, start_date, end_date, mean=None, std=None):
    data = df[station]
    # 判断数据在有效范围内
    feat, label, label_reg = data[0], data[1], data[2]
    referece_start_time = datetime.datetime(2013, 3, 1, 0, 0)
    referece_end_time = datetime.datetime(2017, 2, 28, 0, 0)
    assert (pd.to_datetime(start_date) - referece_start_time).days >= 0
    assert (pd.to_datetime(end_date) - referece_end_time).days <= 0
    assert (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days >= 0
    index_start = (pd.to_datetime(start_date) - referece_start_time).days
    index_end = (pd.to_datetime(end_date) - referece_start_time).days
    feat = feat[index_start: index_end + 1]
    label = label[index_start: index_end + 1]
    label_reg = label_reg[index_start: index_end + 1]
    return data_loader(feat, label, label_reg)


def get_dataloader(df, station, start_time, end_time, batch_size, shuffle=True, mean=None, std=None):
    dataset = create_dataset(df, station, start_time, end_time, mean=mean, std=std)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader


def get_split_time(num_domain=2, mode='pre_process', df=None, station=None, dis_type='coral'):
    spilt_time = {
        '2': [('2013-3-6 0:0', '2015-5-31 23:0'), ('2015-6-2 0:0', '2016-6-30 23:0')]
    }
    if mode == 'pre_process':
        return spilt_time[str(num_domain)]
    if mode == 'tdc':
        return tdc(num_domain, df, station, dis_type=dis_type)
    else:
        print("error in mode")


def load_data_multi_domain(file_path, batch_size=6, station='Changping', number_domain=2, mode='pre_process',
                           dis_type ='coral'):
    # mode: 'tdc', 'pre_process'
    data_file = os.path.join(file_path, "welldata-small.xlsx")
    df_train = pd.read_excel(data_file, sheet_name='train', header=0)
    df_test = pd.read_excel(data_file, sheet_name='test', header=0)
    df_val = pd.read_excel(data_file, sheet_name='val', header=0)

    mean_train, std_train = statistical.get_data_statistic_welldata(df_train)  # mf-从训练集和验证集一起来获取均值和方差
    # 这里调用了TDC算法
    split_time_list = get_split_time(number_domain, mode=mode, df=df, station=station, dis_type=dis_type)

    train_loader_list = []
    for time_temp in split_time_list:
        train_loader = get_dataloader(df, station=station, start_time=time_temp[0],
                                      end_time=time_temp[1], batch_size=batch_size,
                                      mean=mean_train, std=std_train)
        train_loader_list.append(train_loader)

    valid_loader = get_dataloader(df, station=station, start_time='2016-7-2 0:0',
                                  end_time='2016-10-30 23:0', batch_size=batch_size, mean=mean_train,
                                  std=std_train)
    test_loader = get_dataloader(df, station=station, start_time='2016-11-2 0:0',
                                 end_time='2017-2-28 23:0', batch_size=batch_size, mean=mean_train,
                                 std=std_train, shuffle=False)
    return train_loader_list, valid_loader, test_loader
