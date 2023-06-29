# -*- coding: utf-8 -*-
# @Time    : 2023/6/22 0:45
# @Author  : Dreamstar
# @File    : 给一个表格或绘图添加数据-append_data_to_a_table_or_chart.py
# @Link    : https://docs.streamlit.io/knowledge-base/using-streamlit/append-data-table-chart
# @Desc    :
import time

import numpy as np
import streamlit as st

# Get some data.
data = np.random.randn(1, 2)  # 相当于两列数据

# Show the data as a chart.
chart = st.line_chart(data)

for i in range(20):
    # Wait 1 second, so the change is clearer.
    time.sleep(1)

    # Grab some more data.
    data2 = np.random.randn(1, 2)  # 相当于两列数据

    # Append the new data to the existing chart.
    chart.add_rows(data2)

