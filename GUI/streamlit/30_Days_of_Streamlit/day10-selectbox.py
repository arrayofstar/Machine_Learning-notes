# -*- coding: utf-8 -*-
# @Time    : 2023/5/20
# @Author  : Dreamstar
# @File    : day10-selectbox.py
# @Link    : https://30days.streamlit.app/?challenge=Day+10
# @Desc    : st.selectbox 显示一个选择组件


import streamlit as st

st.header('st.selectbox')

option = st.selectbox(
    'What is your favorite color?',
    ('Blue', 'Red', 'Green'))

st.write('Your favorite color is ', option)
