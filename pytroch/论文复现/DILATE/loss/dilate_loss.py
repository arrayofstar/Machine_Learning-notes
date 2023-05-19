import torch

from utils.tools import Timer
from . import soft_dtw
from . import path_soft_dtw


def dilate_loss(outputs, targets, alpha, gamma, device):
    # outputs, targets: shape (batch_size, N_output, 1)
    batch_size, N_output = outputs.shape[0:2]
    loss_shape = 0
    # 0.03 s
    timer = Timer()
    softdtw_batch = soft_dtw.SoftDTWBatch.apply
    # 计算y_pred 和 y_ture的距离矩阵
    D = torch.zeros((batch_size, N_output, N_output)).to(device)
    for k in range(batch_size):
        Dk = soft_dtw.pairwise_distances(targets[k, :, :].view(-1, 1), outputs[k, :, :].view(-1, 1), device=device)
        D[k:k + 1, :, :] = Dk
    print(f'{timer.stop():.6f} sec')
    # 0.22
    timer.start()
    loss_shape = softdtw_batch(D, gamma)
    print(f'{timer.stop():.6f} sec')
    # 1.7s - 这里最慢
    timer.start()
    path_dtw = path_soft_dtw.PathDTWBatch.apply
    path = path_dtw(D, gamma)
    print(f'{timer.stop():.6f} sec')
    # 0.001531 - 还行
    timer.start()
    Omega = soft_dtw.pairwise_distances(torch.range(1, N_output).view(N_output, 1)).to(device)
    loss_temporal = torch.sum(path * Omega) / (N_output * N_output)
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    print(f'{timer.stop():.6f} sec')
    loss = loss_shape
    loss_temporal = None
    return loss, loss_shape, loss_temporal
