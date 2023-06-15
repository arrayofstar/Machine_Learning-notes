# -*- coding: utf-8 -*-
# @Time    : 2023/6/11 12:34
# @Author  : Dreamstar
# @File    : day24-cache-优化性能.py
# @Link    : https://30days.streamlit.app/?challenge=Day24
# @Desc    : st.cache 使得你可以优化 Streamlit 应用的性能。
# @Desc    : 0611-这个功能好像进行了更新，可以使用st.cache_data or st.cache_resource来实现


from time import time

import numpy as np
import pandas as pd
import streamlit as st

st.title('st.cache')

# Using cache
a0 = time()
st.subheader('Using st.cache')

@st.cache_data()
def load_data_a():
  df = pd.DataFrame(
    np.random.rand(2000000, 5),
    columns=['a', 'b', 'c', 'd', 'e']
  )
  return df

st.write(load_data_a())
a1 = time()
st.info(a1-a0)


# Not using cache
b0 = time()
st.subheader('Not using st.cache')

def load_data_b():
  df = pd.DataFrame(
    np.random.rand(2000000, 5),
    columns=['a', 'b', 'c', 'd', 'e']
  )
  return df

st.write(load_data_b())
b1 = time()
st.info(b1-b0)