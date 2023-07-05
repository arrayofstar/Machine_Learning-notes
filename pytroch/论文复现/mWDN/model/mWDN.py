# -*- coding: utf-8 -*-
# @Time    : 2023/7/1 20:12
# @Author  : Dreamstar
# @File    : mWDN.py
# @Link    : 
# @Desc    : 创建一个基于小波分解的网络结构


import sys

import numpy as np
import torch.nn as nn
from scipy import interpolate


class WaveBlock(nn.Module):  # 多级小波网络分解基础
    def __init__(self, seq_len, wavelet=None, alpha=0.5, beta=0.5):
        super(WaveBlock, self).__init__()
        self.alpha = alpha
        self.beta = beta
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
        self.comp_mWDN_H_weight = self.create_W(seq_len, False, is_comp=True)
        self.comp_mWDN_L_weight = self.create_W(seq_len, True, is_comp=True)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AvgPool1d(2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        hp_1 = self.sigmoid(self.mWDN_H(x))
        lp_1 = self.sigmoid(self.mWDN_L(x))
        hp_out = self.pool(hp_1)
        lp_out = self.pool(lp_1)
        all_out = torch.cat((hp_out, lp_out), dim=-1)
        L_loss = torch.norm((self.mWDN_L.weight.data - self.comp_mWDN_L_weight), 2)
        H_loss = torch.norm((self.mWDN_H.weight.data - self.comp_mWDN_H_weight), 2)
        L_H_loss = self.alpha * L_loss + self.beta * H_loss
        return lp_out, hp_out, all_out, L_H_loss

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
    def __init__(self, c_in, c_out, seq_len, hidden_size=256, levels=3, wavelet=None, base_arch='InceptionTimePlus',
                 **kwargs):
        super(mWDN, self).__init__()
        self.levels = levels
        self.mWDN_blocks = nn.ModuleList()
        self.lstm_blocks = nn.ModuleList()
        for i in range(levels):
            self.mWDN_blocks.append(WaveBlock(seq_len // 2 ** i, wavelet=wavelet))
            self.lstm_blocks.append(nn.LSTM(c_in, hidden_size, batch_first=True))
        self.lstm_blocks.append(nn.LSTM(c_in, hidden_size, batch_first=True))
        self.output = nn.Linear(hidden_size * (levels + 1), c_out)

    def forward(self, x):
        x_l = x
        lstm_x_list = []
        loss_list = []
        mWDN_dict = {"mWDN_level": [], "x_l": [], "x_h": []}

        for i in range(self.levels):
            x_l, x_h, out_, loss = self.mWDN_blocks[i](x_l)
            x_l = x_l.permute(0, 2, 1)
            x_h = x_h.permute(0, 2, 1)
            mWDN_dict['mWDN_level'].append(i + 1)
            mWDN_dict['x_l'].append(x_l)
            mWDN_dict['x_h'].append(x_h)
            x_h.retain_grad()  # todo, 不知道会不会梯度累计
            lstm_x_h, _ = self.lstm_blocks[i](x_h)
            lstm_x_list.append(lstm_x_h)
            loss_list.append(loss)
            if i == self.levels - 1:
                x_l.retain_grad()
                lstm_x_l, _ = self.lstm_blocks[i + 1](x_l)  # lstm有两个输出，
                lstm_x_list.append(lstm_x_l)
        lstm_all_cat = torch.tensor([])
        for i in range(len(lstm_x_list)):
            lstm_x_list[i] = lstm_x_list[i].permute(0, 2, 1)
            lstm_x_list[i] = F.interpolate(input=lstm_x_list[i], size=seq_len, mode='linear')
            lstm_x_list[i] = lstm_x_list[i].permute(0, 2, 1)
            lstm_all_cat = torch.cat((lstm_all_cat, lstm_x_list[i]), dim=-1)
        out = self.output(lstm_all_cat)
        return out, np.mean(loss_list), mWDN_dict


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
from torch import as_tensor, Tensor
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


def interp1d_mWDN(input, seq_len): # 插值操作
    x = np.arange(len(input))
    y = input

    xnew = np.linspace(0, 1, seq_len)

    f = interpolate.interp1d(x, y, kind="nearest")
    # f是一个函数，用这个函数就可以找插值点的函数值了：
    ynew = f(xnew)
    return ynew




if __name__ == '__main__':
    device = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

    batch_size = 128
    c_in = 1
    seq_len = 64
    c_out = 1

    input = torch.rand(batch_size, seq_len, c_in).to(device)
    output = torch.rand(batch_size, seq_len, c_out).to(device)
    model = mWDN(c_in, c_out, seq_len).to(device)

    # 前向传播
    output_pred, loss_mWDN, _ = model(input)

    criterion = nn.MSELoss()
    # 反向传播
    # optimizer = optim.Adam(model.parameters())
    # optimizer.zero_grad()
    # loss = criterion(output_pred, output) + loss_mWDN
    # loss.backward()
    # optimizer.step()
    # print(output.shape, loss_mWDN)

    # 重要性分析
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    input = torch.rand(2, seq_len, c_in, requires_grad=True).to(device)
    # model = mWDN(c_in, c_out, seq_len).to(device)  # 旧的

    # 初始化model
    model = mWDN(4, 3, 224).to(device)
    # 重载参数
    checkpoints_path = r''
    model.load_state_dict(torch.load(checkpoints_path))

    # 前向传播
    # output_pred, loss_mWDN, mWDN_details = model(input)  # 单个元素重要性
    #
    # output_pred.sum().backward()
    # input_grad = input.grad
    # input_grad_mean = torch.mean(input_grad, dim=0)
    # # print(input.grad)
    # print(input_grad_mean.shape)

    # # 中间层对模型的重要性
    # input = torch.rand(2, seq_len, c_in).to(device)
    #
    #
    # model = mWDN(c_in, c_out, seq_len).to(device)
    # output_pred, loss_mWDN, mWDN_details = model(input)
    #
    # output_pred.sum().backward()
    # mWDN_grad_x_h_1 = mWDN_details['x_h'][0].grad
    # mWDN_grad_x_h_2 = mWDN_details['x_h'][1].grad
    # mWDN_grad_x_h_3 = mWDN_details['x_h'][2].grad
    # mWDN_grad_x_l_3 = mWDN_details['x_l'][2].grad
    #
    # mWDN_grad_x_h_1_mean = torch.mean(mWDN_details['x_h'][0].grad, dim=0).squeeze().numpy()
    # mWDN_grad_x_h_2_mean = torch.mean(mWDN_details['x_h'][1].grad, dim=0).squeeze().numpy()
    # mWDN_grad_x_h_3_mean = torch.mean(mWDN_details['x_h'][2].grad, dim=0).squeeze().numpy()
    # mWDN_grad_x_l_3_mean = torch.mean(mWDN_details['x_l'][2].grad, dim=0).squeeze().numpy()
    #
    # mWDN_grad_x_h_1_interp = interp1d_mWDN(mWDN_grad_x_h_1_mean, seq_len)
    # mWDN_grad_x_h_2_interp = interp1d_mWDN(mWDN_grad_x_h_2_mean, seq_len)
    # mWDN_grad_x_h_3_interp = interp1d_mWDN(mWDN_grad_x_h_3_mean, seq_len)
    # mWDN_grad_x_l_3_interp = interp1d_mWDN(mWDN_grad_x_l_3_mean, seq_len)
    # df_mWDN = pd.DataFrame([])
    # df_mWDN['x_h_1'] = mWDN_grad_x_h_1_interp
    # df_mWDN['x_h_2'] = mWDN_grad_x_h_2_interp
    # df_mWDN['x_h_3'] = mWDN_grad_x_h_3_interp
    # df_mWDN['x_l_3'] = mWDN_grad_x_l_3_interp

    pass

    # 热力图绘制
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import pandas as pd
    # from sklearn.preprocessing import scale
    # import matplotlib.ticker as ticker

    # sns.set_style("white")  # 绘图风格设置
    # sns.set_context("notebook", font_scale=1.5,
    #                 rc={'axes.labelsize': 17, 'legend.fontsize': 30, 'xtick.labelsize': 15, 'ytick.labelsize': 10})
    # 
    # # df_mWDN.loc[:, :] = scale(df_mWDN.values)  # 数据标准化处理
    # 
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(32, 7), dpi=80)
    # plt.imshow(df_mWDN.T, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    # sns.heatmap(df_mWDN.T, cmap='YlGnBu', linewidths=.10, linecolor='k', square=True)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    # 
    # 
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    # plt.show()
