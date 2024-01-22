# -*- coding: utf-8 -*-
# @Time    : 2023/5/31
# @Author  : Dreamstar
# @File    : day12-checkbox.py
# @Link    : https://30days.streamlit.app/?challenge=Day+12
# @Desc    : st.checkbox 显示一个勾选组件。


import streamlit as st

st.header('st.checkbox')

st.write ('What would you like to order?')

icecream = st.checkbox('Ice cream')
coffee = st.checkbox('Coffee')
cola = st.checkbox('Cola')

if icecream:
     st.write("Great! Here's some more 🍦")

if coffee:
     st.write("Okay, here's some coffee ☕")

if cola:
     st.write("Here you go 🥤")