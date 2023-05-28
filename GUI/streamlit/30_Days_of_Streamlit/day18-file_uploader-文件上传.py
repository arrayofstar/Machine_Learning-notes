#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-03-31 15:45
# @Author  : Dreamstar
# @File    : day18-file_uploader-文件上传.py.py
# @Desc    : https://30days.streamlit.app/?challenge=Day+18

import streamlit as st
import pandas as pd

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