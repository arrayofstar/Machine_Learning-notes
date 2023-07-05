# -*- coding: utf-8 -*-
# @Time    : 2023/7/5 23:38
# @Author  : Dreamstar
# @File    : 数据可视化-plotly.py
# @Link    : https://plotly.com/python/creating-and-updating-figures/
# @Desc    :

import plotly.express as px
import streamlit as st

df = px.data.iris()

fig = (px.scatter(df, x="sepal_width", y="sepal_length", color="species",
            facet_col="species", trendline="ols",
            title="Chaining Multiple Figure Operations With A Plotly Express Figure")
 .update_layout(title_font_size=24)
 .update_xaxes(showgrid=False)
 .update_traces(
     line=dict(dash="dot", width=4),
     selector=dict(type="scatter", mode="lines"))
)




st.plotly_chart(fig)
