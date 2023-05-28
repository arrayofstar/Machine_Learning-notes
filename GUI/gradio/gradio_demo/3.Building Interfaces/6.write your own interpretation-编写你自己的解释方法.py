#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-04-27 11:03
# @Author  : Dreamstar
# @File    : 6.write your own interpretation-编写你自己的解释方法.py
# @Desc    : 编写自己的解释方法

import re

import gradio as gr

male_words, female_words = ["he", "his", "him"], ["she", "hers", "her"]


def gender_of_sentence(sentence):
    male_count = len([word for word in sentence.split() if word.lower() in male_words])
    female_count = len(
        [word for word in sentence.split() if word.lower() in female_words]
    )
    total = max(male_count + female_count, 1)
    return {"male": male_count / total, "female": female_count / total}


# Number of arguments to interpretation function must
# match number of inputs to prediction function
def interpret_gender(sentence):
    result = gender_of_sentence(sentence)
    is_male = result["male"] > result["female"]
    interpretation = []
    for word in re.split("( )", sentence):
        score = 0
        token = word.lower()
        if (is_male and token in male_words) or (not is_male and token in female_words):
            score = 0.2  # 这里的分数和颜色的深浅有关系
        elif (is_male and token in female_words) or (
            not is_male and token in male_words
        ):
            score = -0.2
        interpretation.append((word, score))
    # Output must be a list of lists containing the same number of elements as inputs
    # Each element corresponds to the interpretation scores for the given input
    return [interpretation]


demo = gr.Interface(
    fn=gender_of_sentence,
    inputs=gr.Textbox(value="She went to his house to get her keys."),
    outputs="label",
    interpretation=interpret_gender,
)

demo.launch()
