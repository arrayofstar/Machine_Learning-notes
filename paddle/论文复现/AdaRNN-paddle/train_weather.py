import paddle.nn as nn
import paddle
import paddle.optimizer as optimizer

import os
import argparse
import datetime
import numpy as np

from tqdm import tqdm
from utils import utils
from base.AdaRNN import AdaRNN

import pretty_errors
import dataset.data_process as data_process
import matplotlib.pyplot as plt


def get_args():

    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument('--model_name', default='AdaRNN')
    parser.add_argument('--d_feat', type=int, default=6)

    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--class_num', type=int, default=1)
    parser.add_argument('--pre_epoch', type=int, default=40)  # 20, 30, 50

    # training parameters
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--early_stop', type=int, default=40)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--dw', type=float, default=0.5) # 0.01, 0.05, 5.0
    parser.add_argument('--loss_type', type=str, default='adv')
    parser.add_argument('--station', type=str, default='Dongsi')
    parser.add_argument('--data_mode', type=str,
                        default='tdc')
    parser.add_argument('--num_domain', type=int, default=2)
    parser.add_argument('--len_seq', type=int, default=24)

    # other parameters
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--data_path', default="./data/")
    parser.add_argument('--outdir', default='./outputs')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--log_file', type=str, default='run.log')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--len_win', type=int, default=0)
    args = parser.parse_args()

    return args


def get_model(name='AdaRNN'):
    n_hiddens = [args.hidden_size for i in range(args.num_layers)]
    return AdaRNN(use_bottleneck=True, bottleneck_width=64, n_input=args.d_feat, n_hiddens=n_hiddens,
n_output=args.class_num, dropout=args.dropout, model_type=name, len_seq=args.len_seq,
trans_loss=args.loss_type).cuda()


def main_transfer(args):
    print(args)
    # init path
    output_path = args.outdir + '_' + args.station + '_' + args.model_name + '_weather_' + \
                  args.loss_type + '_' + str(args.pre_epoch) + \
                  '_' + str(args.dw) + '_' + str(args.lr)
    save_model_path = args.model_name + '_' + args.loss_type + \
                      '_' + str(args.dw) + '_' + str(args.lr) + '.pkl'
    utils.dir_exist(output_path)

    pprint('init loaders...')
    train_loader_list, valid_loader, test_loader = data_process.load_weather_data_multi_domain(
        args.data_path, args.batch_size, args.station, args.num_domain, args.data_mode)

    args.log_file = os.path.join(output_path, 'run.log')  #
    pprint('create model...')
    model = get_model(args.model_name)
    num_model = count_parameters(model)
    print('#model params:', num_model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_score = np.inf
    best_epoch, stop_round = 0, 0
    weight_mat, dist_mat = None, None

if __name__ == '__main__':

    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    main_transfer(args)
