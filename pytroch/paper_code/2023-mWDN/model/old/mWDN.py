# -*- coding: utf-8 -*-
# @Time    : 2023/7/1 20:12
# @Author  : Dreamstar
# @File    : mWDN.py
# @Link    : 
# @Desc    : 创建一个基于小波分解的网络结构
# todolist : 1. 将小波分解网络+LSTM 扩展为多级小波分解
#            2. 对训练过程中loss进行提取，参考文献中 3 OPTIMIZATION部分  0.5小时时间
#             W_mWDN1_H = model.mWDN1_H.weight.data
#             W_mWDN1_L = model.mWDN1_L.weight.data
#             W_mWDN2_H = model.mWDN2_H.weight.data
#             W_mWDN2_L = model.mWDN2_L.weight.data
#             L_loss = torch.norm((W_mWDN1_L - model.cmp_mWDN1_L), 2) + torch.norm((W_mWDN2_L - model.cmp_mWDN2_L), 2)
#             H_loss = torch.norm((W_mWDN1_H - model.cmp_mWDN1_H), 2) + torch.norm((W_mWDN2_H - model.cmp_mWDN2_H), 2)
#            3. 重要性分析 - 重点研究

import sys

import numpy as np
import torch.nn as nn


class WaveBlock(nn.Module):  # 多级小波网络分解基础
    def __init__(self, seq_len, wavelet=None):
        super(WaveBlock, self).__init__()
        if wavelet is None:
            self.h_filter = [-0.2304, 0.7148, -0.6309, -0.028, 0.187, 0.0308, -0.0329, -0.0106]
            self.l_filter = [-0.0106, 0.0329, 0.0308, -0.187, -0.028, 0.6309, 0.7148, 0.2304]
        else:
            try:
                import pywt
            except ImportError:
                raise ImportError("You need to either install pywt to run mWDN or set wavelet=None")
            w = pywt.Wavelet(wavelet)
            self.h_filter = w.dec_hi
            self.l_filter = w.dec_lo

        self.mWDN_H = nn.Linear(seq_len, seq_len)
        self.mWDN_L = nn.Linear(seq_len, seq_len)
        self.mWDN_H.weight = nn.Parameter(self.create_W(seq_len, False))
        self.mWDN_L.weight = nn.Parameter(self.create_W(seq_len, True))
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AvgPool1d(2)

    def forward(self, x):
        x = x.permute(0,2,1)
        hp_1 = self.sigmoid(self.mWDN_H(x))
        lp_1 = self.sigmoid(self.mWDN_L(x))
        hp_out = self.pool(hp_1)
        lp_out = self.pool(lp_1)
        all_out = torch.cat((hp_out, lp_out), dim=-1)
        return lp_out, hp_out, all_out

    def create_W(self, P, is_l, is_comp=False):
        if is_l:
            filter_list = self.l_filter
        else:
            filter_list = self.h_filter
        list_len = len(filter_list)
        max_epsilon = np.min(np.abs(filter_list))
        if is_comp:
            weight_np = np.zeros((P, P))
        else:
            weight_np = np.random.randn(P, P) * 0.1 * max_epsilon
        for i in range(0, P):
            filter_index = 0
            for j in range(i, P):
                if filter_index < len(filter_list):
                    weight_np[i][j] = filter_list[filter_index]
                    filter_index += 1
        return tensor(weight_np)


# 示例使用方式：通过定义多级小波分解网络块，获得(batch_size, seq_len, dim) 的多级特征，最后的dim维度代表了所有的分解特征。
class mWDN_example(nn.Module):
    def __init__(self, seq_len, levels=3, wavelet=None, base_arch='InceptionTimePlus', **kwargs):
        super(mWDN, self).__init__()
        self.levels = levels
        self.blocks = nn.ModuleList()
        for i in range(levels):
            self.blocks.append(WaveBlock(seq_len // 2 ** i, wavelet=wavelet))
        # self._model = build_model(base_arch, c_in, c_out, seq_len=seq_len, **kwargs)

    def forward(self, x):
        for i in range(self.levels):
            x, out_ = self.blocks[i](x)
            if i == 0:
                out = out_
            else:
                torch.cat((out, out_), dim=-1)
        # out = self._model(out)
        return out


# 示例使用方式：通过定义多级小波分解网络块，获得(batch_size, seq_len, dim) 的多级特征，最后的dim维度代表了所有的分解特征。
class mWDN(nn.Module):
    def __init__(self, c_in, c_out, seq_len, hidden_size=256, levels=2, wavelet=None, base_arch='InceptionTimePlus', **kwargs):
        super(mWDN, self).__init__()
        self.levels = levels
        self.blocks = nn.ModuleList()
        for i in range(levels):
            self.blocks.append(WaveBlock(seq_len // 2 ** i, wavelet=wavelet))
        # self._model = build_model(base_arch, c_in, c_out, seq_len=seq_len, **kwargs)
        self.lstm_xh1 = nn.LSTM(c_in, hidden_size, batch_first=True)
        self.lstm_xh2 = nn.LSTM(c_in, hidden_size, batch_first=True)
        self.lstm_xl2 = nn.LSTM(c_in, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size*3, c_out)

    def forward(self, x):
        x_l = x
        for i in range(self.levels):
            if i == 0:
                x_l, x_h, out_ = self.blocks[i](x)
                x_l = x_l.permute(0, 2, 1)
                x_h = x_h.permute(0, 2, 1)
                lstm_x_h_1, _ = self.lstm_xh1(x_h)
            else:
                x_l, x_h, out_ = self.blocks[i](x_l)
                x_l = x_l.permute(0, 2, 1)
                x_h = x_h.permute(0, 2, 1)
                lstm_x_h_2, _ = self.lstm_xh2(x_h)
                lstm_x_l_2, _ = self.lstm_xl2(x_h)

        lstm_x_h_1 = lstm_x_h_1.permute(0, 2, 1)
        lstm_x_h_2 = lstm_x_h_2.permute(0, 2, 1)
        lstm_x_l_2 = lstm_x_l_2.permute(0, 2, 1)
        lstm_x_h_1 = F.interpolate(input=lstm_x_h_1, size=seq_len, mode='linear')
        lstm_x_h_2 = F.interpolate(input=lstm_x_h_2, size=seq_len, mode='linear')
        lstm_x_l_2 = F.interpolate(input=lstm_x_l_2, size=seq_len, mode='linear')
        lstm_x_h_1 = lstm_x_h_1.permute(0, 2, 1)
        lstm_x_h_2 = lstm_x_h_2.permute(0, 2, 1)
        lstm_x_l_2 = lstm_x_l_2.permute(0, 2, 1)

        lstm_x_cat = torch.cat((lstm_x_h_1, lstm_x_h_2, lstm_x_l_2), dim=-1)
        out = self.output(lstm_x_cat)
            # out_ = F.interpolate(input=out_, size=64, mode='linear')
            # out = out_ if i == 0 else torch.cat((out, out_), dim=1)
        # out = self._model(out)
        return out


class mWDNBlocks(nn.Module):  # 基础使用方式：这里只包含的mWDN的前置块
    def __init__(self, seq_len, levels=3, wavelet=None):
        super(mWDNBlocks, self).__init__()
        self.levels = levels
        self.blocks = nn.ModuleList()
        self.seq_len = seq_len
        for i in range(levels):
            self.blocks.append(WaveBlock(seq_len // 2 ** i, wavelet=wavelet))

    def forward(self, x):
        for i in range(self.levels):
            x, out_ = self.blocks[i](x)
            out_ = F.interpolate(input=out_, size=64, mode='linear')
            out = out_ if i == 0 else torch.cat((out, out_), dim=1)
        return out


import pandas as pd
import torch
from torch import as_tensor, Tensor, optim
import torch.nn as nn
import torch.nn.functional as F
import numbers
from numpy import array, ndarray


def tensor(x, *rest, **kwargs):
    "Like `torch.as_tensor`, but handle lists too, and can pass multiple vector elements directly."
    if len(rest): x = (x,) + rest
    # There was a Pytorch bug in dataloader using num_workers>0. Haven't confirmed if fixed
    # if isinstance(x, (tuple,list)) and len(x)==0: return tensor(0)
    res = (x if isinstance(x, Tensor)
           else torch.tensor(x, **kwargs) if isinstance(x, (tuple, list, numbers.Number))
    else _array2tensor(x, **kwargs) if isinstance(x, ndarray)
    else as_tensor(x.values, **kwargs) if isinstance(x, (pd.Series, pd.DataFrame))
    else _array2tensor(array(x), **kwargs))
    if res.dtype is torch.float64: return res.float()
    return res


def _array2tensor(x, requires_grad=False, pin_memory=False, **kwargs):
    if x.dtype == np.uint16: x = x.astype(np.float32)
    # windows default numpy int dtype is int32, while torch tensor default int dtype is int64
    # https://github.com/numpy/numpy/issues/9464
    if sys.platform == "win32" and x.dtype == int: x = x.astype(np.int64)
    t = torch.as_tensor(x, **kwargs)
    t.requires_grad_(requires_grad)
    if pin_memory: t.pin_memory()
    return t


if __name__ == '__main__':
    device = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

    batch_size = 128
    c_in = 4
    seq_len = 32
    c_out = 3

    input = torch.rand(batch_size, seq_len, c_in).to(device)
    output = torch.rand(batch_size, seq_len, c_out).to(device)
    model = mWDN(c_in, c_out, seq_len).to(device)

    # 前向传播
    output_pred = model(input)

    criterion = nn.MSELoss()
    # 反向传播
    optimizer = optim.Adam(model.parameters())
    optimizer.zero_grad()
    loss = criterion(output_pred, output)
    loss.backward()
    optimizer.step()

    print(output.shape)
