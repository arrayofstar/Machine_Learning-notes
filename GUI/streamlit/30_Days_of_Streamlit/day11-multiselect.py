# -*- coding: utf-8 -*-
# @Time    : 2023/5/31
# @Author  : Dreamstar
# @File    : day11-multiselect.py
# @Link    : https://30days.streamlit.app/?challenge=Day+11
# @Desc    : st.multiselect 显示一个多选组件。


import streamlit as st

st.header('st.multiselect')

options = st.multiselect(
     'What are your favorite colors',
     ['Green', 'Yellow', 'Red', 'Blue'],
     ['Yellow', 'Red'])

st.write('You selected:', options)