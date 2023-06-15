#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-03-31 15:09
# @Author  : Dreamstar
# @File    : day16-customize_theme-定制主题.py.py
# @Link    : https://30days.streamlit.app/?challenge=Day+16
# @Desc    : 通过调整 config.toml 中的选项来自定义应用的主题，这个配置文件应当被放在与应用并行的 .streamlit 文件夹内。

import streamlit as st

st.title('Customizing the theme of Streamlit apps')

st.write('Contents of the `.streamlit/config.toml` file of this app')

st.code("""
[theme]
primaryColor="#F39C12"
backgroundColor="#2E86C1"
secondaryBackgroundColor="#AED6F1"
textColor="#FFFFFF"
font="monospace"
""")

number = st.sidebar.slider('Select a number:', 0, 10, 5)
st.write('Selected number from slider widget is:', number)