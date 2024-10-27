from efficient_kan.kan import KAN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import os
from tqdm import tqdm
import pandas as pd


# Load Data
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


file_path = r"C:\Users\ADMIN\Desktop"
train_df = pd.read_csv(os.path.join(file_path, 'train.csv'), na_values=-99999)
train_df = train_df.drop("wellnum", axis=1)
val_df = pd.read_csv(os.path.join(file_path, 'vali.csv'), na_values=-99999)
val_df = val_df.drop("wellnum", axis=1)
test_df = pd.read_csv(os.path.join(file_path, 'test.csv'), na_values=-99999)
test_df = test_df.drop("wellnum", axis=1)

trainset = Well_Dataset(train_df.drop('AC', axis=1).values, train_df['AC'].values, train_df.index)
valset = Well_Dataset(val_df.drop('AC', axis=1).values, val_df['AC'].values, train_df.index)
testset = Well_Dataset(test_df.drop('AC', axis=1).values, test_df['AC'].values, train_df.index)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


# Define model
model = KAN([4, 64, 64, 1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# Define loss
criterion = nn.MSELoss()
metric = nn.L1Loss()
for epoch in range(100):
    # Train
    model.train()
    with tqdm(trainloader) as pbar:
        for i, (feature, labels, index) in enumerate(pbar):
            feature = feature.to(device)
            optimizer.zero_grad()
            output = model(feature)
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            mae_loss = metric(output, labels.to(device))
            pbar.set_postfix(loss=loss.item(), mae_loss=mae_loss.item(), lr=optimizer.param_groups[0]['lr'])

    # Validation
    model.eval()
    val_loss = 0
    val_mae_loss = 0
    with torch.no_grad():
        for feature, labels, index in valloader:
            feature = feature.to(device)
            output = model(feature)
            val_loss += criterion(output, labels.to(device)).item()
            val_mae_loss += metric(output, labels.to(device)).item()
    val_loss /= len(valloader)
    val_mae_loss /= len(valloader)

    # Update learning rate
    scheduler.step()

    print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val MAE Loss: {val_mae_loss}"
    )

test_loss = 0
test_mae_loss = 0
with torch.no_grad():
    for feature, labels, index in valloader:
        feature = feature.to(device)
        output = model(feature)
        test_loss += criterion(output, labels.to(device)).item()
        test_mae_loss += metric(output, labels.to(device)).item()
test_loss /= len(valloader)
test_mae_loss /= len(valloader)

print(f"Test Loss: {test_loss}, Test MAE Loss: {test_mae_loss}")