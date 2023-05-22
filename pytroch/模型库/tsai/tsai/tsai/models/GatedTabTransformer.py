# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/065_models.GatedTabTransformer.ipynb.

# %% auto 0
__all__ = ['GatedTabTransformer']

# %% ../../nbs/065_models.GatedTabTransformer.ipynb 4
import torch.nn as nn

from .TabTransformer import TabTransformer
from .gMLP import gMLP


# %% ../../nbs/065_models.GatedTabTransformer.ipynb 5
class _TabMLP(nn.Module):
    def __init__(self, classes, cont_names, c_out, d_model, mlp_d_model, mlp_d_ffn, mlp_layers):
        super().__init__()
        seq_len = d_model * len(classes) + len(cont_names)
        self.mlp = gMLP(1, c_out, seq_len, d_model=mlp_d_model, d_ffn=mlp_d_ffn, depth=mlp_layers)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        return self.mlp(x)


class GatedTabTransformer(TabTransformer):
    def __init__(self, classes, cont_names, c_out, column_embed=True, add_shared_embed=False, shared_embed_div=8, embed_dropout=0.1, drop_whole_embed=False, 
                 d_model=32, n_layers=6, n_heads=8, d_k=None, d_v=None, d_ff=None, res_attention=True, attention_act='gelu', res_dropout=0.1, norm_cont=True,
                 mlp_d_model=32, mlp_d_ffn=64, mlp_layers=4):

        super().__init__(classes, cont_names, c_out, column_embed=column_embed, add_shared_embed=add_shared_embed, shared_embed_div=shared_embed_div,
                         embed_dropout=embed_dropout, drop_whole_embed=drop_whole_embed, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_k=d_k,
                         d_v=d_v, d_ff=d_ff, res_attention=res_attention, attention_act=attention_act, res_dropout=res_dropout, norm_cont=norm_cont)

        self.mlp = _TabMLP(classes, cont_names, c_out, d_model, mlp_d_model, mlp_d_ffn, mlp_layers)
