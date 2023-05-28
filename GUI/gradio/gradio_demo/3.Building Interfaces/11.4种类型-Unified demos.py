#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-04-27 11:38
# @Author  : Dreamstar
# @File    : 11.4种类型-Unified demos.py
# @Desc    : 统一输入和输出接口

import gradio as gr
from transformers import pipeline

generator = pipeline('text-generation', model = 'gpt2')

def generate_text(text_prompt):
  response = generator(text_prompt, max_length = 30, num_return_sequences=5)
  return response[0]['generated_text']

textbox = gr.Textbox()

demo = gr.Interface(generate_text, textbox, textbox)

demo.launch()
