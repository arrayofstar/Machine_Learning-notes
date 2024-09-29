# ！/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/9/29 14:31
# @Author  : Dreamstar
# @File    : demo.py
# @Tool    : PyCharm
# @Desc    : 
# @Link    :


import torch

from torchvision import models
from gpu_mem_track import MemTracker

device = torch.device('cuda:0')

gpu_tracker = MemTracker()         # define a GPU tracker

gpu_tracker.track()                     # run function between the code line where uses GPU
cnn = models.vgg19(pretrained=True).features.to(device).eval()
gpu_tracker.track()                     # run function between the code line where uses GPU

dummy_tensor_1 = torch.randn(30, 3, 512, 512).float().to(device)  # 30*3*512*512*4/1024/1024 = 90.00M
dummy_tensor_2 = torch.randn(40, 3, 512, 512).float().to(device)  # 40*3*512*512*4/1024/1024 = 120.00M
dummy_tensor_3 = torch.randn(60, 3, 512, 512).float().to(device)  # 60*3*512*512*4/1024/1024 = 180.00M

gpu_tracker.track()

dummy_tensor_4 = torch.randn(120, 3, 512, 512).float().to(device)  # 120*3*512*512*4/1024/1024 = 360.00M
dummy_tensor_5 = torch.randn(80, 3, 512, 512).float().to(device)  # 80*3*512*512*4/1024/1024 = 240.00M

gpu_tracker.track()

dummy_tensor_4 = dummy_tensor_4.cpu()
dummy_tensor_2 = dummy_tensor_2.cpu()
gpu_tracker.clear_cache() # or torch.cuda.empty_cache()

gpu_tracker.track()