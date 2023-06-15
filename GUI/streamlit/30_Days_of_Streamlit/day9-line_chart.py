# -*- coding: utf-8 -*-
# @Time    : 2023/5/20
# @Author  : Dreamstar
# @File    : day9-line_chart.py
# @Link    : https://30days.streamlit.app/?challenge=Day+9
# @Desc    : st.line_chart 显示一个折线图。适合于很多“画个图看看”的场景，但不易调节样式。


import numpy as np
import pandas as pd
import streamlit as st

st.header('Line chart')

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])

st.line_chart(chart_data)
