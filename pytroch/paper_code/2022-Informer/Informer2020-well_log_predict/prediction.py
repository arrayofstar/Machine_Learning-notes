# -*- coding: utf-8 -*-
# @Time    : 2022/10/12 21:18
# @Author  : Dreamstar
# @File    : prediction.py
# @Desc    :

import os
import torch
from utils.tools import dotdict
from exp.exp_informer import Exp_Informer

args = dotdict()

args.model = 'informer'  # model of experiment, options: [informer, informerstack, informerlight(TBD)]

args.data = 'my_data'  # data
args.root_path = './data/'  # root path of data file
args.data_path = 'welldata_100'  # data file
args.features = 'MS'  # forecasting task, options:[M, S, MS];
# M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
args.target = 'AC'  # target feature in S or MS task
args.freq = None  # freq for time features encoding,
# options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly],
# you can also use more detailed freq like 15min or 3h
args.checkpoints = './checkpoints'  #  location of model checkpoints

args.seq_len = 33  # input sequence length of Informer encoder
args.label_len = 16  # start token length of Informer decoder
args.pred_len = 1  # prediction sequence length
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

args.enc_in = 7  # encoder input size
args.dec_in = 7  # decoder input size
args.c_out = 7  # output size
args.factor = 5  # probsparse attn factor
args.d_model = 512  # dimension of model
args.n_heads = 8  # num of heads
args.e_layers = 2  # num of encoder layers
args.d_layers = 1  # num of decoder layers
args.d_ff = 2048  # dimension of fcn in model
args.dropout = 0.05  # dropout
args.attn = 'prob'  # attention used in encoder, options:[prob, full]
args.embed = 'learned'  # time features encoding, options:[timeF, fixed, learned]
args.activation = 'gelu'  # activation
args.distil = True  # whether to use distilling in encoder
args.output_attention = False  # whether to output attention in ecoder
args.mix = True
args.padding = 0

args.batch_size = 32
args.learning_rate = 0.0001
args.loss = 'mse'
args.lradj = 'type1'
args.use_amp = False  # whether to use automatic mixed precision training

# args.num_workers = 0
args.itr = 1
# args.train_epochs = 6
# args.patience = 3
# args.des = 'exp'

args.use_gpu = True if torch.cuda.is_available() else False
args.gpu = 0

args.use_multi_gpu = False
args.devices = '0,1,2,3'

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# Set augments by using data name
data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'well_log_data_s': {'data': 'well_log_data_s.xlsx', 'T': 'AC', 'M': [4, 4, 4],
                        'S': [1, 1, 1], 'MS': [4, 4, 1]},
    'my_data': {'data': 'welldata_100', 'T': 'AC', 'M': [4, 4, 4], 'S': [1, 1, 1], 'MS': [4, 4, 1],
                    'do_predict': True},
}

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]
args.detail_freq = args.freq

print('Args in experiment:')
print(args)

Exp = Exp_Informer

# set saved model path
# setting = 'informer_well_log_data_s_featuresMS_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048attnprob_fc5_eblearned_dtTrue_mxTrue_test_0'

ii = 0
setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model,
                args.data, args.features,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor,
                args.embed, args.distil, args.mix, args.des, ii)

path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')

exp = Exp(args)

exp.predict(setting, True)