# -*- coding: utf-8 -*-
# @Time    : 2024/7/12 下午6:36
# @Author  : Dreamstar
# @File    : 1.chat-gpt.py
# @Desc    : 
# @Link    :


import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUMC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(GRUMC, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mc_dropout=False):
        gru_out, _ = self.gru(x)  # [batch_size, seq_length, hidden_dim]
        if mc_dropout:
            gru_out = self.dropout(gru_out)  # 在测试时启用Dropout
        else:
            gru_out = F.dropout(gru_out, p=self.dropout.p, training=self.training)  # 在训练时启用Dropout
        output = self.linear(gru_out)  # [batch_size, seq_length, output_dim]
        return output

# 假设输入数据
input_dim = 4
hidden_dim = 64
output_dim = 1
dropout_rate = 0.5

model = GRUMC(input_dim, hidden_dim, output_dim, dropout_rate)

# 输入特征 (batch_size, seq_length, feature_dim)
x = torch.randn(128, 33, 4)

# 使用 MC Dropout 进行预测
num_samples = 100
# model.eval()  # 设置模型为评估模式
preds = []
with torch.no_grad():  # 禁用自动求导
    for _ in range(num_samples):
        preds.append(model(x, mc_dropout=True).unsqueeze(0))  # 加一个维度以用于后续的拼接

preds = torch.cat(preds, dim=0)  # (num_samples, batch_size, seq_length, output_dim)

# 计算预测的均值和标准差
mean_preds = preds.mean(dim=0)  # (batch_size, seq_length, output_dim)
std_preds = preds.std(dim=0)  # (batch_size, seq_length, output_dim)

print("Mean predictions:", mean_preds)
print("Standard deviation of predictions:", std_preds)

