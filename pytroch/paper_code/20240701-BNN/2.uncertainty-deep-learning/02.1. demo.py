# -*- coding: utf-8 -*-
# @Time    : 2024/7/12 下午7:21
# @Author  : Dreamstar
# @File    : 02.1. demo.py
# @Desc    : 
# @Link    :

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


device = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

class TransformerMC(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim, dropout_rate=0.5):
        super(TransformerMC, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Transformer Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Dropout layer for MC Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Linear layer to produce the final output
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, mc_dropout=False):
        x = x.permute(1, 0, 2)  # (batch_size, seq_length, input_dim) -> (seq_length, batch_size, input_dim)
        x = self.transformer_encoder(x)
        if mc_dropout and self.training:
            x = self.dropout(x)  # Apply Dropout during inference
        x = x.permute(1, 0, 2)  # (seq_length, batch_size, input_dim) -> (batch_size, seq_length, input_dim)
        output = self.linear(x)  # (batch_size, seq_length, output_dim)
        return output


# 生成示例数据
batch_size = 128
seq_length = 33
input_dim = 4
output_dim = 1
dropout_rate = 0.5


# 创建随机数据
x = torch.randn(batch_size, seq_length, input_dim).to(device)  # (batch_size, seq_length, input_dim)
y = torch.randn(batch_size, seq_length, output_dim).to(device)  # (batch_size, seq_length, output_dim)

# 归一化数据
x_mean, x_std = x.mean(), x.std()
y_mean, y_std = y.mean(), y.std()
x = (x - x_mean) / x_std
y = (y - y_mean) / y_std


# 初始化模型
hidden_dim = 64
num_heads = 4
num_layers = 3

model = TransformerMC(input_dim, hidden_dim, num_heads, num_layers, output_dim, dropout_rate).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 训练模型
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    output = model(x, mc_dropout=False)  # 训练时不使用 Dropout
    y_true = torch.Tensor(y).to(device)
    loss = criterion(output, y_true)  # 只计算最后一个时间步的损失
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")


# MC Dropout 进行预测
num_samples = 100
# model.eval()  # 设置模型为评估模式
preds = []

with torch.no_grad():  # 禁用自动求导
    for _ in range(num_samples):
        preds.append(model(x, mc_dropout=True))  # 使用 Dropout 进行 MC 预测

# 将 MC Dropout 的预测结果堆叠起来
preds = torch.stack(preds).cpu()  # (num_samples, batch_size, seq_length, output_dim)

# 计算预测的均值和标准差
mean_preds = preds.mean(dim=0).cpu()  # (batch_size, seq_length, output_dim)
std_preds = preds.std(dim=0).cpu()  # (batch_size, seq_length, output_dim)

# 计算 95% 和 75% 置信区间
conf_95_upper = mean_preds + 1.96 * std_preds  # 95% 置信区间上界
conf_95_lower = mean_preds - 1.96 * std_preds  # 95% 置信区间下界
conf_75_upper = mean_preds + 1.15 * std_preds  # 75% 置信区间上界
conf_75_lower = mean_preds - 1.15 * std_preds  # 75% 置信区间下界
conf_25_upper = mean_preds + 0.674 * std_preds  # 25% 置信区间上界
conf_25_lower = mean_preds - 0.674 * std_preds  # 25% 置信区间下界


# 选择第一个样本进行可视化
x_test_flat = torch.linspace(-0.5, 1, seq_length).numpy()
y_pred = mean_preds[0, :, 0].numpy()  # 选择第一个批次的预测结果

# 计算置信区间
conf_95_upper_np = conf_95_upper[0, :, 0].numpy()
conf_95_lower_np = conf_95_lower[0, :, 0].numpy()
conf_75_upper_np = conf_75_upper[0, :, 0].numpy()
conf_75_lower_np = conf_75_lower[0, :, 0].numpy()
conf_25_upper_np = conf_25_upper[0, :, 0].numpy()
conf_25_lower_np = conf_25_lower[0, :, 0].numpy()

plt.figure(figsize=(12, 6))
plt.plot(x_test_flat, y_pred, label='Mean Predictions')
plt.fill_between(x_test_flat, conf_95_lower_np, conf_95_upper_np, alpha=0.3, color='blue', label='95% Confidence Interval')
plt.fill_between(x_test_flat, conf_75_lower_np, conf_75_upper_np, alpha=0.5, color='orange', label='75% Confidence Interval')
plt.fill_between(x_test_flat, conf_25_lower_np, conf_25_upper_np, alpha=0.7, color='green', label='25% Confidence Interval')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Prediction')
plt.title('MC Dropout Predictions with Uncertainty and Confidence Intervals')
plt.grid()
plt.show()
