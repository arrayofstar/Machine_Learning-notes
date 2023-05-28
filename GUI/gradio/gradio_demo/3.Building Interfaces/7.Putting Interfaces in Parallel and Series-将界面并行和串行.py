#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-04-27 11:10
# @Author  : Dreamstar
# @File    : 7.Putting Interfaces in Parallel and Series-将界面并行和串行.py
# @Desc    : 这里介绍了并行计算和串行计算的方式
import gradio as gr

# generator1 = gr.Interface.load("huggingface/gpt2")
# generator2 = gr.Interface.load("huggingface/EleutherAI/gpt-neo-2.7B")
# generator3 = gr.Interface.load("huggingface/EleutherAI/gpt-j-6B")
#
# gr.Parallel(generator1, generator2, generator3).launch()  # 并行的方法 同时运行


generator = gr.load("huggingface/gpt2")
translator = gr.load("huggingface/t5-small")

gr.Series(generator, translator).launch()  # 串行的方法 依次运行

# this demo generates text, then translates it to German, and outputs the final result.
