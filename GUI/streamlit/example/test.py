# -*- coding: utf-8 -*-
# @Time    : 2023/6/22 16:01
# @Author  : Dreamstar
# @File    : functional_test.py
# @Link    : 
# @Desc    : 用于测试不同的streamlit代码
import time

import numpy as np
import streamlit as st

st.write("# 用于提前测试不同streamlit代码情况")

"streamlit中的按钮状态会随着另一个按钮的状态改变而改变，" \
"如果希望streamlit能够记住安装按下的状态的话，就需要使用session_state来记录状态"

# Initialization
if 'click_button_1' not in st.session_state:
    st.session_state['click_button_1'] = False
if 'my_slider' not in st.session_state:
    st.session_state['my_slider'] = 0

def click_button1_callback():
    st.session_state['click_button_1'] = True

click_button1 = st.button("测试一下", key="test1", on_click=click_button1_callback)

st.write(st.session_state['click_button_1'])

click_button2 = st.button("测试一下", key="test2")

st.write(click_button2)

"如果提前设定的某个输入控件的内容状态的话，它的值将不会被你所改变：尝试一下改变下面滑动条的值"
st.session_state.my_slider = 7

slider = st.slider(
    label='My Slider', min_value=1,
    max_value=10, value=5, key='my_slider')

st.write(slider)

uploader_file = st.file_uploader("数据上传", type=['csv','xlsx','xls'], accept_multiple_files=False, key=None,
                                 help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
st.write(uploader_file)

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

