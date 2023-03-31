'''
https://30days.streamlit.app/?challenge=Day+9
st.write允许将文本和参数写入Streamlight应用程序。
'''


import streamlit as st
import pandas as pd
import numpy as np

st.header('Line chart')

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])

st.line_chart(chart_data)
