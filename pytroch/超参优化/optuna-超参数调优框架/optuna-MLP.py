# -*- coding: utf-8 -*-
# @Time    : 2022/8/12 11:00
# @Author  : Dreamstar
# @File    : optuna-MLP.py
# @Desc    : 关于的
# pyuic5 MainWindow.ui -o MainWindow.py
# PySide6-uic MainWindow.ui -o MainWindow.py

import time

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable  # 获取变量
import optuna


class MyDataset(TensorDataset):
    def __init__(self, df_data):
        # 对数据进行预处理
        wanted_columns = [col for col in df_data.columns.tolist() if col not in "AC"]
        features = df_data[wanted_columns].values
        labels = df_data['AC'].values
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        super(MyDataset, self).__init__(features_tensor, labels_tensor)


class MLP(nn.Module):

    def __init__(self, hidden_layer, activation_func):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(4, hidden_layer)  # 假设输入特征为10
        self.act = activation_func  # 使用提供的激活函数
        self.fc2 = nn.Linear(hidden_layer, 1)  # 输出层

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze()
        return x


def get_activation_func(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError("Unsupported activation function")


# 定义早停类
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, loss_func=None, delta=0, path='checkpoint.pth.tar'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pth.tar'
        """
        self.patience = patience
        self.verbose = verbose
        self.loss_func = loss_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss |{self.loss_func}| decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  '
                  f'Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# 自定义回调函数显示进度
class ProgressCallback():
    def __init__(self, n_total):
        self.n_total = n_total
        self.start_time = time.time()
        self.current_iteration = 0

    def __call__(self, study, trial):
        self.current_iteration += 1
        time_taken = time.time() - self.start_time
        best_score = study.best_value
        # n_trials = len(study.trials)
        progress = (self.current_iteration / self.n_total) * 100
        print(f"Iteration {self.current_iteration}/{self.n_total} - Best score: {best_score:.4f} - "
              f"Progress: {progress:.2f}% - Time elapsed: {time_taken:.2f} seconds")


def train(batch_size, learning_rate, lossfunc, opt, hidden_layer, activation_func_name, weightdk, momentum):  # 选出一些超参数
    print("================== Model Info ==================")
    print(f"batch_size: | {batch_size}")
    print(f"learning_rate: | {learning_rate}")
    print(f"lossfunc: | {lossfunc}")
    print(f"optimizer: | {opt}")
    print(f"model_hidden_layer: | {hidden_layer}")
    print(f"activation_func_name: | {activation_func_name}")
    print(f"train_para_weightdk: | {weightdk}")
    print(f"train_para_momentum: | {momentum}")
    print("=====================================================")
    train_set = pd.read_csv(r'C:\Users\ADMIN\Desktop\train.csv')
    train_set = train_set.drop('wellnum', axis=1)
    vali_set = pd.read_csv(r'C:\Users\ADMIN\Desktop\vali.csv')
    vali_set = vali_set.drop('wellnum', axis=1)
    num_epochs = 100
    early_stopping = EarlyStopping(patience=10, verbose=True, loss_func=lossfunc)

    train_dataset = MyDataset(train_set)
    vali_dataset = MyDataset(vali_set)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    vali_loader = DataLoader(vali_dataset, batch_size=batch_size, shuffle=True)

    activation_func = get_activation_func(activation_func_name)
    # 创建CNN模型， 并设置损失函数及优化器
    model = MLP(hidden_layer, activation_func).cuda()
    # print(model)
    if lossfunc == 'MSE':
        criterion = nn.MSELoss().cuda()
    elif lossfunc == 'MAE':
        criterion = nn.L1Loss()

    if opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weightdk)
    elif opt == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weightdk, momentum=momentum)
    # 训练过程
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0], data[1]
            inputs = Variable(inputs).float().cuda()
            labels = Variable(labels).float().cuda()
            # 前向传播
            out = model(inputs)
            # 可以考虑加正则项
            opt_train_loss = criterion(out, labels)
            optimizer.zero_grad()
            opt_train_loss.backward()
            optimizer.step()
            train_loss = train_loss + opt_train_loss.item()
        train_loss /= len(train_loader)
        # 验证模式
        model.eval()
        # 返回测试集合上的MAE
        vali_loss = 0
        with torch.no_grad():
            for data in vali_loader:
                inputs, labels = data
                inputs, labels = inputs.float().cuda(), labels.float().cuda()
                out = model(inputs)
                vali_loss += criterion(out, labels).item()
        vali_loss /= len(vali_loader)
        # 输出训练进度
        if (epoch + 1) % 10 == 0:  # 每10个批次输出一次
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                f'train_loss: {train_loss:.4f}, vali_loss: {vali_loss:.4f}')
        early_stopping(vali_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return vali_loss


def objective(trail):
    batchsize = trail.suggest_categorical('batchsize', [256, 512])
    lr = trail.suggest_float('lr', 1e-4, 1e-2, log=True)  # 使用log=True以更好地在范围内搜索
    opt = trail.suggest_categorical('opt', ['Adam', 'RMSprop'])
    hidden_layer = trail.suggest_int('hiddenlayer', 20, 1200)
    activefunc = trail.suggest_categorical('active', ['relu', 'sigmoid', 'tanh'])
    weightdekey = trail.suggest_float('weight_dekay', 0, 1, step=0.01)
    momentum = trail.suggest_float('momentum', 0, 1, step=0.01)
    lossfunc = 'MAE'  # MSE
    loss = train(batchsize, lr, lossfunc, opt, hidden_layer, activefunc, weightdekey, momentum)
    return loss


if __name__ == '__main__':
    n_trials = 10
    st = time.time()
    study = optuna.create_study(study_name='search_best_MLP', direction='minimize')
    # progress_callback = ProgressCallback(n_total=n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)  # , callbacks=progress_callback
    print(study.best_params)
    print(study.best_trial)
    print(study.best_trial.value)
    print(time.time() - st)

    optuna.visualization.plot_param_importances(study).show()  # 参数重要性
    optuna.visualization.plot_optimization_history(study).show()  # 优化历史
    optuna.visualization.plot_slice(study).show()  # 单因素比较
    optuna.visualization.plot_parallel_coordinate(study).show()

