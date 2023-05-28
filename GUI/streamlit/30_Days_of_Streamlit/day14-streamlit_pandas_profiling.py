#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-03-31 14:56
# @Author  : Dreamstar
# @File    : day14-streamlit_pandas_profiling.py
# @Desc    : https://30days.streamlit.app/?challenge=Day+14

import streamlit as st
import pandas as pd
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

st.header('`streamlit_pandas_profiling`')

df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')

pr = df.profile_report()
st_profile_report(pr)