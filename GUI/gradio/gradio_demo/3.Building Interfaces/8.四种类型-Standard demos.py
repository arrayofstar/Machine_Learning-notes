#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-04-27 11:15
# @Author  : Dreamstar
# @File    : 8.4种类型-Standard demos.py
# @Desc    : 标准的模式，需要输入和输出接口，输入数据经过函数处理后在输出接口显示
#            要创建一个同时包含输入和输出组件的演示，只需在Interface（）中设置输入和输出参数的值。下面是一个简单图像过滤器的示例演示：

import gradio as gr
import numpy as np


def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img


demo = gr.Interface(sepia, gr.Image(shape=(200, 200)), "image")
demo.launch()
