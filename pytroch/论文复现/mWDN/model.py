import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import ToVariable


class Wavelet_LSTM(nn.Module):
    def __init__(self, seq_len, hidden_size, output_size, device='cpu'):
        super(Wavelet_LSTM, self).__init__()
        self.device = device
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.mWDN1_H = nn.Linear(seq_len, seq_len)
        self.mWDN1_L = nn.Linear(seq_len, seq_len)
        self.mWDN2_H = nn.Linear(int(seq_len / 2), int(seq_len / 2))
        self.mWDN2_L = nn.Linear(int(seq_len / 2), int(seq_len / 2))
        self.a_to_x = nn.AvgPool1d(2)
        self.sigmoid = nn.Sigmoid()
        self.lstm_xh1 = nn.LSTM(1, hidden_size, batch_first=True)
        self.lstm_xh2 = nn.LSTM(1, hidden_size, batch_first=True)
        self.lstm_xl2 = nn.LSTM(1, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

        self.l_filter = [-0.0106, 0.0329, 0.0308, -0.187, -0.028, 0.6309, 0.7148, 0.2304]
        self.h_filter = [-0.2304, 0.7148, -0.6309, -0.028, 0.187, 0.0308, -0.0329, -0.0106]

        self.cmp_mWDN1_H = ToVariable(self.create_W(seq_len, False, is_comp=True)).to(self.device)
        self.cmp_mWDN1_L = ToVariable(self.create_W(seq_len, True, is_comp=True)).to(self.device)
        self.cmp_mWDN2_H = ToVariable(self.create_W(int(seq_len / 2), False, is_comp=True)).to(self.device)
        self.cmp_mWDN2_L = ToVariable(self.create_W(int(seq_len / 2), True, is_comp=True)).to(self.device)

        self.mWDN1_H.weight = torch.nn.Parameter(ToVariable(self.create_W(seq_len, False)))
        self.mWDN1_L.weight = torch.nn.Parameter(ToVariable(self.create_W(seq_len, True)))
        self.mWDN2_H.weight = torch.nn.Parameter(ToVariable(self.create_W(int(seq_len / 2), False)))
        self.mWDN2_L.weight = torch.nn.Parameter(ToVariable(self.create_W(int(seq_len / 2), True)))

    def forward(self, input, h1, c1, h2, c2, h3, c3):
        input = input.view(input.shape[0], input.shape[1])  # 实际上只有一个序列维度
        ah_1 = self.sigmoid(self.mWDN1_H(input))
        al_1 = self.sigmoid(self.mWDN1_L(input))
        xh_1 = self.a_to_x(ah_1.view(ah_1.shape[0], 1, -1))
        xl_1 = self.a_to_x(al_1.view(al_1.shape[0], 1, -1))

        ah_2 = self.sigmoid(self.mWDN2_H(xl_1))
        al_2 = self.sigmoid(self.mWDN2_L(xl_1))

        xh_2 = self.a_to_x(ah_2)
        xl_2 = self.a_to_x(al_2)

        xh_1 = xh_1.transpose(1, 2)
        xh_2 = xh_2.transpose(1, 2)
        xl_2 = xl_2.transpose(1, 2)

        level1_lstm, (h1, c1) = self.lstm_xh1(xh_1, (h1, c1))
        level2_lstm_h, (h2, c2) = self.lstm_xh2(xh_2, (h2, c2))
        level2_lstm_l, (h3, c3) = self.lstm_xl2(xl_2, (h3, c3))

        output = self.output(torch.cat((level1_lstm, level2_lstm_h, level2_lstm_l), 1))
        # output = output.view(-1,1)
        return output, h1, c1, h2, c2, h3, c3

    def init_state(self, batch_size):
        h1 = Variable(torch.zeros(1, batch_size, self.hidden_size)).double().to(self.device)
        c1 = Variable(torch.zeros(1, batch_size, self.hidden_size)).double().to(self.device)

        h2 = Variable(torch.zeros(1, batch_size, self.hidden_size)).double().to(self.device)
        c2 = Variable(torch.zeros(1, batch_size, self.hidden_size)).double().to(self.device)

        h3 = Variable(torch.zeros(1, batch_size, self.hidden_size)).double().to(self.device)
        c3 = Variable(torch.zeros(1, batch_size, self.hidden_size)).double().to(self.device)
        return h1, c1, h2, c2, h3, c3

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
        return weight_np


class Wavelet_GRU(nn.Module):  # mf - 根据测井实际需求进行修改
    def __init__(self, seq_len, hidden_size, output_size, device='cpu'):
        super(Wavelet_GRU, self).__init__()
        self.device = device
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.mWDN1_H = nn.Linear(seq_len, seq_len)
        self.mWDN1_L = nn.Linear(seq_len, seq_len)
        self.mWDN2_H = nn.Linear(int(seq_len / 2), int(seq_len / 2))
        self.mWDN2_L = nn.Linear(int(seq_len / 2), int(seq_len / 2))
        self.a_to_x = nn.AvgPool1d(2)
        self.sigmoid = nn.Sigmoid()
        self.gru_xh1 = nn.GRU(1, hidden_size, batch_first=True)
        self.gru_xh2 = nn.GRU(1, hidden_size, batch_first=True)
        self.gru_xl2 = nn.GRU(1, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

        self.l_filter = [-0.0106, 0.0329, 0.0308, -0.187, -0.028, 0.6309, 0.7148, 0.2304]
        self.h_filter = [-0.2304, 0.7148, -0.6309, -0.028, 0.187, 0.0308, -0.0329, -0.0106]

        # self.cmp_mWDN1_H = ToVariable(self.create_W(seq_len, False, is_comp=True)).to(self.device)
        # self.cmp_mWDN1_L = ToVariable(self.create_W(seq_len, True, is_comp=True)).to(self.device)
        # self.cmp_mWDN2_H = ToVariable(self.create_W(int(seq_len / 2), False, is_comp=True)).to(self.device)
        # self.cmp_mWDN2_L = ToVariable(self.create_W(int(seq_len / 2), True, is_comp=True)).to(self.device)

        self.mWDN1_H.weight = torch.nn.Parameter(ToVariable(self.create_W(seq_len, False)))
        self.mWDN1_L.weight = torch.nn.Parameter(ToVariable(self.create_W(seq_len, True)))
        self.mWDN2_H.weight = torch.nn.Parameter(ToVariable(self.create_W(int(seq_len / 2), False)))
        self.mWDN2_L.weight = torch.nn.Parameter(ToVariable(self.create_W(int(seq_len / 2), True)))

    def forward(self, input):
        batch_size, seq_len, fea_len = input.shape[0], input.shape[1], input.shape[2]
        h1, h2, h3 = self.init_state(batch_size)

        input = input.view(batch_size, seq_len)  # 实际上只有一个序列维度
        ah_1 = self.sigmoid(self.mWDN1_H(input))
        al_1 = self.sigmoid(self.mWDN1_L(input))
        xh_1 = self.a_to_x(ah_1.view(ah_1.shape[0], 1, -1))
        xl_1 = self.a_to_x(al_1.view(al_1.shape[0], 1, -1))

        ah_2 = self.sigmoid(self.mWDN2_H(xl_1))
        al_2 = self.sigmoid(self.mWDN2_L(xl_1))

        xh_2 = self.a_to_x(ah_2)
        xl_2 = self.a_to_x(al_2)

        xh_1 = xh_1.transpose(1, 2)
        xh_2 = xh_2.transpose(1, 2)
        xl_2 = xl_2.transpose(1, 2)

        level1_gru, h1 = self.gru_xh1(xh_1, h1)
        level2_gru_h, h2 = self.gru_xh2(xh_2, h2)
        level2_gru_l, h3 = self.gru_xl2(xl_2, h3)

        output = self.output(torch.cat((level1_gru, level2_gru_h, level2_gru_l), 1))
        # output = output.view(-1,1)
        return output, h1, h2, h3

    def init_state(self, batch_size):
        h1 = Variable(torch.zeros(1, batch_size, self.hidden_size)).double().to(self.device)
        h2 = Variable(torch.zeros(1, batch_size, self.hidden_size)).double().to(self.device)
        h3 = Variable(torch.zeros(1, batch_size, self.hidden_size)).double().to(self.device)
        return h1, h2, h3

    def create_W(self, P, is_l, is_comp=False):
        if is_l:
            filter_list = self.l_filter
        else:
            filter_list = self.h_filter

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
        return weight_np


if __name__ == '__main__':
    """
    这里始终还是有一个问题，就是这个模型和方法针对的还是单个曲线的多级分解，并不能用于多种曲线，这是一个问题
    """
    print("start")
    input = torch.rand(size=(32, 12, 1), dtype=torch.float64)
    model = Wavelet_LSTM(seq_len=12, hidden_size=32, output_size=1)
    model = model.double()
    h1, c1, h2, c2, h3, c3 = model.init_state(input.shape[0])
    output, h1, c1, h2, c2, h3, c3 = model(input, h1, c1, h2, c2, h3, c3)
    print(output.size())

    input = torch.rand(size=(32, 12, 1), dtype=torch.float64)
    model = Wavelet_GRU(seq_len=12, hidden_size=32, output_size=1)
    model = model.double()
    output, h1, h2, h3 = model(input)
    print(output.size())

