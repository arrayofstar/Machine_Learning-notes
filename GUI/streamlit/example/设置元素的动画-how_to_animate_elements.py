# -*- coding: utf-8 -*-
# @Time    : 2023/6/22 0:37
# @Author  : Dreamstar
# @File    : 设置元素的动画-how_to_animate_elements.py
# @Link    : https://docs.streamlit.io/knowledge-base/using-streamlit/animate-elements
# @Desc    :
import time

import numpy as np
import streamlit as st

progress_bar = st.progress(0)
status_text = st.empty()
line_data = np.random.randn(10, 1)  # 一个长度为10的线
chart = st.line_chart(line_data)

for i in range(100):
    # Update progress bar.
    progress_bar.progress(i + 1)

    new_rows = np.random.randn(10, 1)  # 一个长度为10的线

    # Update status text.
    status_text.text(
        'The latest random number is: %s' % new_rows[-1])

    # Append data to the chart.
    chart.add_rows(new_rows)

    # Pretend we're doing some computation that takes time.
    time.sleep(0.1)

status_text.text('Done!')
st.balloons()