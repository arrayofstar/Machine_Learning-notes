# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/034_models.ResNetPlus.ipynb.

# %% auto 0
__all__ = ['ResBlockPlus', 'ResNetPlus']

# %% ../../nbs/034_models.ResNetPlus.ipynb 3
from fastai.layers import *

from .layers import *


# %% ../../nbs/034_models.ResNetPlus.ipynb 4
class ResBlockPlus(Module):
    def __init__(self, ni, nf, ks=[7, 5, 3], coord=False, separable=False, bn_1st=True, zero_norm=False, sa=False, se=None, act=nn.ReLU, act_kwargs={}):
        self.convblock1 = ConvBlock(
            ni, nf, ks[0], coord=coord, separable=separable, bn_1st=bn_1st, act=act, act_kwargs=act_kwargs)
        self.convblock2 = ConvBlock(
            nf, nf, ks[1], coord=coord, separable=separable, bn_1st=bn_1st, act=act, act_kwargs=act_kwargs)
        self.convblock3 = ConvBlock(
            nf, nf, ks[2], coord=coord, separable=separable, zero_norm=zero_norm, act=None)
        self.se = SEModule1d(
            nf, reduction=se, act=act) if se and nf//se > 0 else noop
        self.sa = SimpleSelfAttention(nf, ks=1) if sa else noop
        self.shortcut = BN1d(ni) if ni == nf else ConvBlock(
            ni, nf, 1, coord=coord, act=None)
        self.add = Add()
        self.act = act(**act_kwargs)

        self._init_cnn(self)

    def _init_cnn(self, m):
        if getattr(self, 'bias', None) is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(self.weight)
        for l in m.children():
            self._init_cnn(l)

    def forward(self, x):
        res = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.se(x)
        x = self.sa(x)
        x = self.add(x, self.shortcut(res))
        x = self.act(x)
        return x


@delegates(ResBlockPlus.__init__)
class ResNetPlus(nn.Sequential):
    def __init__(self, c_in, c_out, seq_len=None, nf=64, sa=False, se=None, fc_dropout=0., concat_pool=False,
                 flatten=False, custom_head=None, y_range=None, **kwargs):

        resblock1 = ResBlockPlus(c_in,   nf,     se=se,   **kwargs)
        resblock2 = ResBlockPlus(nf,     nf * 2, se=se,   **kwargs)
        resblock3 = ResBlockPlus(nf * 2, nf * 2, sa=sa, **kwargs)
        backbone = nn.Sequential(resblock1, resblock2, resblock3)
        
        self.head_nf = nf * 2
        if flatten:
            assert seq_len is not None, "you need to pass seq_len when flatten=True"
            self.head_nf *= seq_len
        if custom_head is not None:
            if isinstance(custom_head, nn.Module): head = custom_head
            else: head = custom_head(self.head_nf, c_out, seq_len)
        else:
            head = self.create_head(self.head_nf, c_out, flatten=flatten,
                                         concat_pool=concat_pool, fc_dropout=fc_dropout, y_range=y_range)
        super().__init__(OrderedDict([('backbone', backbone), ('head', head)]))
            
    def create_head(self, nf, c_out, flatten=False, concat_pool=False, fc_dropout=0., y_range=None, **kwargs):
        layers = [Flatten()] if flatten else []
        if concat_pool:
            nf = nf * 2
        layers = [GACP1d(1) if concat_pool else GAP1d(1)]
        if fc_dropout:
            layers += [nn.Dropout(fc_dropout)]
        layers += [nn.Linear(nf, c_out)]
        if y_range:
            layers += [SigmoidRange(*y_range)]
        return nn.Sequential(*layers)
