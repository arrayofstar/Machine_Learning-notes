# -*- coding: utf-8 -*-
# @Time    : 2024/10/31 12:26
# @Author  : Dreamstar
# @File    : check_cuda.py
# @Link    : 
# @Desc    : 测试pytroch对应的cuda是否可以正常运行

import torch

def check_gpu():
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print("CUDA is available. Testing GPU...")
        # 在GPU上创建一个随机张量
        tensor = torch.rand((2, 3)).cuda()
        # 将张量移回CPU并打印
        tensor = tensor.cpu()
        print("Tensor created on GPU:", tensor)
        print("GPU test passed.")
    else:
        print("CUDA is not available. GPU is not working or not installed properly.")

# 调用函数
check_gpu()