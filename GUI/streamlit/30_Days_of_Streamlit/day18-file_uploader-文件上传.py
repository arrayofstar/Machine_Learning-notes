#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-03-31 15:45
# @Author  : Dreamstar
# @File    : day18-file_uploader-文件上传.py.py
# @Link    : https://30days.streamlit.app/?challenge=Day+18
# @Desc    : 一个上传文件的组件
#            默认情况下，上传的文件大小不能超过 200MB。你可以在通过 server.maxUploadSize 选项对其进行配置。

import pandas as pd
import streamlit as st

st.title('st.file_uploader')

st.subheader('Input CSV')
uploaded_file = st.file_uploader("Choose a file", label='1234')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader('DataFrame')
    st.write(df)
    st.subheader('Descriptive Statistics')
    st.write(df.describe())
else:
    st.info('☝️ Upload a CSV file')