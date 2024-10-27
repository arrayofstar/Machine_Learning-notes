# 仅单文件 - 链接来源为 https://github.com/timeseriesAI/tsai/blob/main/tsai/models/mWDN.py

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/052_models.mWDN.ipynb.

# %% auto 0
__all__ = ['WaveBlock', 'mWDN', 'mWDNBlocks', 'mWDNPlus']

from InceptionTimePlus import *
# %% ../../nbs/052_models.mWDN.ipynb 3
from ts_imports import *
from utils import build_model


# %% ../../nbs/052_models.mWDN.ipynb 4
# This is an unofficial PyTorch implementation by Ignacio Oguiza - oguiza@timeseriesAI.co based on:

# Wang, J., Wang, Z., Li, J., & Wu, J. (2018, July). Multilevel wavelet decomposition network for interpretable time series analysis. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 2437-2446).
# No official implementation found


class WaveBlock(Module):
    def __init__(self, c_in, c_out, seq_len, wavelet=None):
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
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AvgPool1d(2)

    def forward(self, x):
        hp_1 = self.sigmoid(self.mWDN_H(x))
        lp_1 = self.sigmoid(self.mWDN_L(x))
        hp_out = self.pool(hp_1)
        lp_out = self.pool(lp_1)
        all_out = torch.cat((hp_out, lp_out), dim=-1)
        return lp_out, all_out

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


# %% ../../nbs/052_models.mWDN.ipynb 5
class mWDN(Module):
    def __init__(self, c_in, c_out, seq_len, levels=3, wavelet=None, base_arch=InceptionTimePlus, **kwargs):
        self.levels = levels
        self.blocks = nn.ModuleList()
        for i in range(levels):
            self.blocks.append(WaveBlock(c_in, c_out, seq_len // 2 ** i, wavelet=wavelet))
        self._model = build_model(base_arch, c_in, c_out, seq_len=seq_len, **kwargs)

    def forward(self, x):
        for i in range(self.levels):
            x, out_ = self.blocks[i](x)
            if i == 0:
                out = out_ if i == 0 else torch.cat((out, out_), dim=-1)
        out = self._model(out)
        return out


class mWDNBlocks(Module):
    def __init__(self, c_in, c_out, seq_len, levels=3, wavelet=None):
        self.levels = levels
        self.blocks = nn.ModuleList()
        for i in range(levels):
            self.blocks.append(WaveBlock(c_in, c_out, seq_len // 2 ** i, wavelet=wavelet))

    def forward(self, x):
        for i in range(self.levels):
            x, out_ = self.blocks[i](x)
            if i == 0:
                out = out_ if i == 0 else torch.cat((out, out_), dim=-1)
        return out


class mWDNPlus(nn.Sequential):
    def __init__(self, c_in, c_out, seq_len, levels=3, wavelet=None, base_model=None, base_arch=InceptionTimePlus,
                 **kwargs):
        if base_model is None:
            base_model = build_model(base_arch, c_in, c_out, seq_len=seq_len, **kwargs)
        blocks = mWDNBlocks(c_in, c_out, seq_len, levels=levels, wavelet=wavelet)
        backbone = nn.Sequential(blocks, base_model.backbone)
        super().__init__(OrderedDict([('backbone', backbone), ('head', base_model.head)]))
        self.head_nf = base_model.head_nf


if __name__ == '__main__':
    pass
