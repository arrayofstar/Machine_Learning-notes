import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader


def create_synthetic_dataset(N, N_input, N_output, sigma):
    # N: number of samples in each split (train, test)
    # N_input: import of time steps in input series
    # N_output: import of time steps in output series
    # sigma: standard deviation of additional noise
    X = []
    breakpoints = []
    for k in range(2 * N):
        serie = np.array([sigma * random.random() for i in range(N_input + N_output)])
        i1 = random.randint(1, 10)
        i2 = random.randint(10, 18)
        j1 = random.random()
        j2 = random.random()
        interval = abs(i2 - i1) + random.randint(-3, 3)
        serie[i1:i1 + 1] += j1
        serie[i2:i2 + 1] += j2
        serie[i2 + interval:] += (j2 - j1)
        X.append(serie)
        breakpoints.append(i2 + interval)
    X = np.stack(X)
    breakpoints = np.array(breakpoints)
    return X[0:N, 0:N_input], X[0:N, N_input:N_input + N_output], X[N:2 * N, 0:N_input], X[N:2 * N,
                                                                                         N_input:N_input + N_output], breakpoints[
                                                                                                                      0:N], breakpoints[
                                                                                                                            N:2 * N]


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, X_input, X_target, breakpoints):
        super(SyntheticDataset, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.X_input = X_input
        self.X_target = X_target
        self.breakpoints = breakpoints

    def __len__(self):
        return (self.X_input).shape[0]

    def __getitem__(self, idx):
        x = self.X_input[idx, :, np.newaxis]
        y = self.X_target[idx, :, np.newaxis]
        bkp = self.breakpoints[idx]
        return x, y, bkp

if __name__ == '__main__':
    # parameters
    batch_size = 100
    N = 500
    N_input = 20
    N_output = 20
    sigma = 0.01
    gamma = 0.01

    # Load synthetic dataset
    X_train_input, X_train_target, X_test_input, X_test_target, train_bkp, test_bkp = create_synthetic_dataset(N,
                                                                                                               N_input,
                                                                                                               N_output,
                                                                                                               sigma)

    dataset_train = SyntheticDataset(X_train_input, X_train_target, train_bkp)
    dataset_test = SyntheticDataset(X_test_input, X_test_target, test_bkp)
    trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)

    for x, y, bkp in trainloader:
        pass

