import torch
from kan import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import os
from tqdm import tqdm
import pandas as pd

# 设置设备和训练精度
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
print(torch.__version__)


# 读取数据 - Load Data
class Well_Dataset(Dataset):
    def __init__(self, feature, label, index, transform=None):
        # 判断特征、标签、回归标签长度一致
        assert len(feature) == len(label), "Features and labels must have the same length."

        self.feature = feature
        self.label = label
        self.index = index

        self.T = transform
        self.feature = torch.tensor(self.feature, dtype=torch.float32)
        label = torch.tensor(self.label, dtype=torch.float32)
        self.label = label.view(-1, 1)  # 或者 target = target.reshape(-1, 1)

    def __getitem__(self, index):
        sample, target, df_idx = self.feature[index], self.label[index], self.index[index]
        if self.T:
            return self.T(sample), target, df_idx
        else:
            return sample, target, df_idx

    def __len__(self):
        return len(self.feature)


def get_data():
    file_path = r"C:\Users\ADMIN\Desktop"
    train_df = pd.read_csv(os.path.join(file_path, 'train.csv'), na_values=-99999)
    train_df = train_df.drop("wellnum", axis=1)
    vali_df = pd.read_csv(os.path.join(file_path, 'vali.csv'), na_values=-99999)
    vali_df = vali_df.drop("wellnum", axis=1)
    # test_df = pd.read_csv(os.path.join(file_path, 'test.csv'), na_values=-99999)
    # trainset = Well_Dataset(train_df.drop('AC', axis=1).values, train_df['AC'].values, train_df.index)
    # valset = Well_Dataset(val_df.drop('AC', axis=1).values, val_df['AC'].values, train_df.index)
    #
    # trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    # valloader = DataLoader(valset, batch_size=64, shuffle=False)

    train_feature = train_df.drop('AC', axis=1).values[:15000]
    train_label = train_df['AC'].values[:15000]
    vali_feature = vali_df.drop('AC', axis=1).values[:5000]
    vali_label = vali_df['AC'].values[:5000]

    dataset = {"train_input": torch.tensor(train_feature, dtype=torch.float32).to(device),
               "train_label": torch.tensor(train_label, dtype=torch.float32).to(device),
               "test_input": torch.tensor(vali_feature, dtype=torch.float32).to(device),
               "test_label": torch.tensor(vali_label, dtype=torch.float32).to(device),
               }
    return dataset


dataset = get_data()
# Define model
model = KAN(width=[4, 5, 1], grid=5, k=3, seed=0, device=device)

# plot KAN at initialization
model(dataset['train_input'])
plt = model.plot(beta=100)
plt.savefig(f'mf_test/1.init_mode.png', bbox_inches="tight", dpi=400)

# train the model
model.train(dataset, opt="LBFGS", steps=10, lamb=0.01, lamb_entropy=10., device=device)
plt = model.plot()
plt.savefig(f'mf_test/2.train_model_1.png', bbox_inches="tight", dpi=400)
print("初始化训练")

model.prune()
plt = model.plot(mask=True)
plt.savefig(f'mf_test/3.prune_model.png', bbox_inches="tight", dpi=400)

model = model.prune()
model(dataset['train_input'])
plt = model.plot()
plt.savefig(f'mf_test/4.train_model_2.png', bbox_inches="tight", dpi=400)

# 减枝后再训练
model.train(dataset, opt="LBFGS", steps=100, device=device)
plt = model.plot()
plt.savefig(f'mf_test/5.train_model_LBFGS_2.png', bbox_inches="tight", dpi=400)

# 自动尝试多种组合
mode = "auto"  # "manual"
if mode == "manual":
    # manual mode
    model.fix_symbolic(0, 0, 0, 'sin')
    model.fix_symbolic(0, 1, 0, 'x^2')
    model.fix_symbolic(1, 0, 0, 'exp')
elif mode == "auto":
    # automatic mode
    lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']
    model.auto_symbolic(lib=lib)

model.train(dataset, opt="LBFGS", steps=100, device=device)
plt = model.plot()
plt.savefig(f'mf_test/6.train_model_last.png', bbox_inches="tight", dpi=400)

print(model.symbolic_formula()[0][0])
