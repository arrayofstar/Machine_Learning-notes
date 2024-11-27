# -*- coding: utf-8 -*-
# @Time    : 2024/11/26 17:07
# @Author  : Dreamstar
# @File    : train.py
# @Link    : 
# @Desc    : 根据《动手学深度学习》中的代码，来复现一下BERT的预训练、微调和预测过程
import numpy as np
import torch
from torch import nn
from lightning.fabric import Fabric
from dataloader import load_data_wiki
from model import BERTModel
from tqdm import tqdm
from d2l import torch as d2l


#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # 前向传播
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # 计算遮蔽语言模型损失
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1))
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # 计算下一句子预测任务的损失
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l


def pretrain(batch_size, max_len, num_steps=50):
    # 1. 数据
    train_dataloader, vocab = load_data_wiki(batch_size, max_len)
    # 2. 模型
    devices = d2l.try_all_gpus()
    model = BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
                        ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                        num_layers=2, dropout=0.2, key_size=128, query_size=128,
                        value_size=128, hid_in_features=128, mlm_in_features=128,
                        nsp_in_features=128).to(devices[0])
    print(model)
    # 3. 训练
    fabric = Fabric(accelerator="cuda")  # precision='bf16-mixed'
    fabric.launch()
    loss = nn.CrossEntropyLoss()

    vocab_size = len(vocab)

    # model = nn.DataParallel(model, device_ids=devices).to(devices[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    step, timer = 0, d2l.Timer()
    # animator = d2l.Animator(xlabel='step', ylabel='loss',
    #                         xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    metric = d2l.Accumulator(4)  # metric中一共可以存4个变量
    mlm_loss_list = []
    nsp_loss_list = []

    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for batch in tqdm(train_dataloader):
            tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y = batch
            optimizer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                model, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            fabric.backward(l)
            optimizer.step()
            mlm_loss_list.append(mlm_l.item() / tokens_X.shape[0])
            nsp_loss_list.append(nsp_l.item() / tokens_X.shape[0])
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            # animator.add(step + 1, (metric[0] / metric[3], metric[1] / metric[3]))
        mlm_loss = np.average(mlm_loss_list) * 1000
        nsp_loss = np.average(nsp_loss_list) * 1000
        step += 1
        if step == num_steps:
            num_steps_reached = True
            break
        print(f'epoch: {step} - MLM loss {mlm_loss:.3f}, NSP loss {nsp_loss:.3f}')

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on {str(devices)}')
    return model


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    batch_size, max_len = 512, 64
    num_steps = 200
    model = pretrain(batch_size=512, max_len=64)
    torch.save(model.state_dict(), 'checkpoint.pth')
