#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-04-27 10:01
# @Author  : Dreamstar
# @File    : 3.Live Interfaces-实时界面.py
# @Desc    : 实现实时刷新的功能，通过在界面中设置live=True，可以使界面自动刷新。现在，一旦用户输入发生变化，界面就会重新计算。

import gradio as gr


def calculator(num1, operation, num2):
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        return num1 / num2


demo = gr.Interface(
    calculator,
    [
        "number",
        gr.Radio(["add", "subtract", "multiply", "divide"]),
        "number"
    ],
    "number",
    live=True,
)
demo.launch()

