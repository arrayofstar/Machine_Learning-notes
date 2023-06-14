import torch

from load_data import load_data
from model import Wavelet_LSTM
from train import train, test


def main():
    data_path = "./Data/GasPrice.csv"
    P = 12  # sequence length
    step = 3  # ahead predict steps

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_train, Y_train, X_test, Y_test, data_df_combined_clean = load_data(data_path, P=P, step=step)
    print(X_train.shape)
    print(Y_train.shape)

    model = Wavelet_LSTM(seq_len=P, hidden_size=32, output_size=1, device=device)
    model = model.double()
    model = model.to(device)

    train(model, X_train, Y_train, epochs=20, device=device)
    test(model, X_test, Y_test, data_df_combined_clean, device=device)


if __name__ == "__main__":
    main()
