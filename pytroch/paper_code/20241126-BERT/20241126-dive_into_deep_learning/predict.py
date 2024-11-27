# -*- coding: utf-8 -*-
# @Time    : 2024/11/26 22:26
# @Author  : Dreamstar
# @File    : predict.py
# @Link    : 
# @Desc    :

import numpy as np
import torch
from torch import nn
from dataloader import load_data_wiki
from model import BERTModel
from d2l import torch as d2l

def get_bert_encoding(vocab, net, tokens_a, tokens_b=None, devices=None,):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X

def predict(batch_size, max_len):
    # 1. 数据
    train_iter, vocab = load_data_wiki(batch_size, max_len, mode='train')

    # 2. 模型
    devices = d2l.try_all_gpus()
    model = BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
                      ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                      num_layers=2, dropout=0.2, key_size=128, query_size=128,
                      value_size=128, hid_in_features=128, mlm_in_features=128,
                      nsp_in_features=128).to(devices[0])
    print(model)

    best_model_path = 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path, weights_only=True))  # weights_only=True

    tokens_a = ['a', 'crane', 'is', 'flying']
    encoded_text = get_bert_encoding(vocab, model, tokens_a, devices=devices)  # 词元：'<cls>','a','crane','is','flying','<sep>'
    encoded_text_cls = encoded_text[:, 0, :]
    encoded_text_crane = encoded_text[:, 2, :]
    print(encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3])

    tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
    encoded_pair = get_bert_encoding(vocab, model, tokens_a, tokens_b, devices=devices)
    # 词元：'<cls>','a','crane','driver','came','<sep>','he','just', 'left','<sep>'
    encoded_pair_cls = encoded_pair[:, 0, :]
    encoded_pair_crane = encoded_pair[:, 2, :]
    print(encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3])


if __name__ == '__main__':
    batch_size, max_len = 512, 64
    predict(batch_size, max_len)
