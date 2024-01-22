# -*- coding: utf-8 -*-
# @Time    : 2023/5/20
# @Author  : Dreamstar
# @File    : day3-button.py
# @Link    : https://30days.streamlit.app/?challenge=Day+3
# @Desc    : Streamlit中的按钮功能


import streamlit as st

st.header('st.button')

if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')
