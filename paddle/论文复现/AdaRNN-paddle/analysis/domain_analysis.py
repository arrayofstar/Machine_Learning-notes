# -*- coding: utf-8 -*-
# @Time    : 2022/10/22 17:27
# @Author  : Dreamstar
# @File    : TDC.py
# @Desc    :
import datetime
import math

import pandas as pd
import torch

from analysis.loss_transfer import TransferLoss


def tdc(num_domain, data, station, dis_type = 'coral'):
    # mf-备注，这里的时间其实可以根据训练数据和验证数据的时间开始和结束来获取
    start_time = datetime.datetime.strptime(
        '2013-03-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    end_time = datetime.datetime.strptime(
        '2016-06-30 23:00:00', '%Y-%m-%d %H:%M:%S')
    num_day = (end_time - start_time).days
    split_n = 10
    data = data[station]
    feat = data[0][0:num_day]
    feat = torch.tensor(feat, dtype=torch.float32)
    feat_shape_1 = feat.shape[1]
    feat = feat.reshape(-1, feat.shape[2])
    feat = feat.cuda()
    # num_day_new = feat.shape[0]

    selected = [0, 10]
    candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    start = 0

    if num_domain in [2, 3, 5, 7, 10]:
        while len(selected) - 2 < num_domain - 1:
            distance_list = []
            for can in candidate:
                selected.append(can)
                selected.sort()
                dis_temp = 0
                for i in range(1, len(selected) - 1):
                    for j in range(i, len(selected) - 1):
                        index_part1_start = start + math.floor(selected[i - 1] / split_n * num_day) * feat_shape_1
                        index_part1_end = start + math.floor(selected[i] / split_n * num_day) * feat_shape_1
                        feat_part1 = feat[index_part1_start: index_part1_end]
                        index_part2_start = start + math.floor(selected[j] / split_n * num_day) * feat_shape_1
                        index_part2_end = start + math.floor(selected[j + 1] / split_n * num_day) * feat_shape_1
                        feat_part2 = feat[index_part2_start:index_part2_end]
                        criterion_transder = TransferLoss(loss_type=dis_type, input_dim=feat_part1.shape[1])
                        dis_temp += criterion_transder.compute(feat_part1, feat_part2)
                distance_list.append(dis_temp)
                selected.remove(can)
            can_index = distance_list.index(max(distance_list))
            selected.append(candidate[can_index])
            candidate.remove(candidate[can_index])
        selected.sort()
        res = []
        for i in range(1, len(selected)):
            if i == 1:
                sel_start_time = start_time + datetime.timedelta(days=int(num_day / split_n * selected[i - 1]), hours=0)
            else:
                sel_start_time = start_time + datetime.timedelta(days=int(num_day / split_n * selected[i - 1]) + 1,
                                                                 hours=0)
            sel_end_time = start_time + datetime.timedelta(days=int(num_day / split_n * selected[i]), hours=23)
            sel_start_time = datetime.datetime.strftime(sel_start_time, '%Y-%m-%d %H:%M')
            sel_end_time = datetime.datetime.strftime(sel_end_time, '%Y-%m-%d %H:%M')
            res.append((sel_start_time, sel_end_time))
        return res
    else:
        print("error in number of domain")