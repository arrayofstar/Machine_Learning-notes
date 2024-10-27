# -*- coding: utf-8 -*-
# @project: Machine_Learning-notes
# @Time    : 2024/1/23 15:46
# @Author  : Dreamstar
# @File    : TCN-well_logging.py
# @Desc    : 用于测井曲线重构的模型


import sys

import torch

sys.path.append("../../")  # 设置导入库的搜索路径
from torch import nn
from tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, fea_dim, label_dim, fea_len=33, label_len=33, num_channels=[64, 32, 16], kernel_size=2, dropout=0.2):
        """
        用书测井曲线重构的TCN模型 - 简化版
        :param fea_dim: 输入数据维度
        :param label_dim: 输出数据维度
        :param num_channels: TCN网络中多层的通道数，作用在序列的维度上。 示例：[32,64,32]
        :param kernel_size: 卷积核的长度，代表的感受野的大小。
        :param dropout: TCN中的dropout层（两层）
        """

        super(TCN, self).__init__()
        self.encoder = nn.Linear(fea_len, num_channels[0])  # 编码器
        self.tcn = TemporalConvNet(num_channels[0], num_channels, kernel_size=kernel_size, dropout=dropout)
        self.decoder = nn.Linear(num_channels[-1], label_len)  # 解码器
        self.linear_feature = nn.Linear(fea_dim, label_dim)  # 对于特征维度的操作
        # self.decoder.weight = self.encoder.weight  # 暂时不需要这个操作
        # self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # 输入数据的维度是 (batch_size=128, seq_len=33, fea_dim=4), 输出数据的维度是 (batch_size, label_len, label_dim)
        x = x.transpose(1, 2)  # 将seq_len和fea_dim对调 (128, 33, 4) -> (128, 4, 33)
        x = self.encoder(x)  # (128, 4, 33) -> (128, 4, 64)
        x = self.tcn(x.transpose(1, 2))  # (128, 4, 64) -> (128, 64, 4) -> (128, 16, 4)
        x = self.decoder(x.transpose(1, 2))  # (128, 16, 4) -> (128, 4, 16) -> (128, 4, 33)
        x = self.linear_feature(x.transpose(1, 2))  # (128, 4, 33) -> (128. 33, 4) -> (128, 33, 1)
        return x


if __name__ == '__main__':
    # 测井曲线重构目标 - todo需要把
    input = torch.rand(128, 33, 4)
    output = torch.rand(128, 33, 1)
    model = TCN(fea_dim=input.size()[2], label_dim=output.size()[2], fea_len=33, label_len=33, num_channels=[64, 32, 16])
    output_hat = model(input)
    print(output_hat.size())
