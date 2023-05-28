#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-04-27 11:18
# @Author  : Dreamstar
# @File    : 9.4种类型-Output-only demo.py
# @Desc    : 只需要输出数据和执行函数


import time

import gradio as gr


def fake_gan():
    time.sleep(1)
    images = [
            "https://img1.baidu.com/it/u=721855496,2511992242&fm=253&fmt=auto&app=138&f=PNG?w=781&h=500",
            "https://img2.baidu.com/it/u=495514904,2671907137&fm=253&fmt=auto&app=138&f=JPEG?w=2154&h=458",
            "https://www.leixue.com/uploads/2020/08/PyTorch.png%21760",
    ]
    return images


demo = gr.Interface(
    fn=fake_gan,
    inputs=None,
    outputs=gr.Gallery(label="Generated Images", show_label=False,).style(columns=3,  object_fit='contain'),
    title="FD-GAN",
    description="This is a fake demo of a GAN. In reality, the images are randomly chosen from Unsplash.",
)

demo.launch()
