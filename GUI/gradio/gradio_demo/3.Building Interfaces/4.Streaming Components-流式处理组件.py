#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-04-27 10:01
# @Author  : Dreamstar
# @File    : 4.Streaming Components-流式处理组件.py
# @Desc    : 实时读取摄像头数据并呈现


import gradio as gr
import numpy as np


def flip(im):
    return np.flipud(im)


demo = gr.Interface(
    flip,
    gr.Image(source="webcam", streaming=True),
    "image",
    live=True
)
demo.launch()
