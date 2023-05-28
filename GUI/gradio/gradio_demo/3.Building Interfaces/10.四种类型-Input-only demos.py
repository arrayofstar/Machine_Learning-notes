#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-04-27 11:28
# @Author  : Dreamstar
# @File    : 10.4种类型-Input-only demos.py
# @Desc    : 仅需要输入接口和处理函数


import random
import string

import gradio as gr


def save_image_random_name(image):
    random_string = ''.join(random.choices(string.ascii_letters, k=20)) + '.png'
    image.save(random_string)
    print(f"Saved image to {random_string}!")


demo = gr.Interface(
    fn=save_image_random_name,
    inputs=gr.Image(type="pil"),
    outputs=None,
)
demo.launch()
