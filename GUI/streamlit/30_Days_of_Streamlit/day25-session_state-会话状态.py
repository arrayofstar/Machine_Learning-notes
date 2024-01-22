# -*- coding: utf-8 -*-
# @Time    : 2023/6/11 12:40
# @Author  : Dreamstar
# @File    : day25-session_state-会话状态.py
# @Link    : https://30days.streamlit.app/?challenge=Day25
# @Desc    : 会话状态（Session State）是一个在同一会话的不同次重新运行间共享变量的方法。
# @Desc    : 除了能够存储和保留状态，Streamlit 还提供了使用回调函数更改状态的支持。


import streamlit as st

st.title('st.session_state')

def lbs_to_kg():
  st.session_state.kg = st.session_state.lbs/2.2046
def kg_to_lbs():
  st.session_state.lbs = st.session_state.kg*2.2046

st.header('Input')
col1, spacer, col2 = st.columns([2,1,2])
with col1:
  pounds = st.number_input("Pounds:", key = "lbs", on_change = lbs_to_kg)
with col2:
  kilogram = st.number_input("Kilograms:", key = "kg", on_change = kg_to_lbs)

st.header('Output')
st.write("st.session_state object:", st.session_state)