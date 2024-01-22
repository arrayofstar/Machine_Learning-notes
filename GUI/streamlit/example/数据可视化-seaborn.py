# -*- coding: utf-8 -*-
# @Time    : 2023/7/4 21:40
# @Author  : Dreamstar
# @File    : 数据可视化.py
# @Link    : https://github.com/apress/beginners-guide-streamlit-python
# @Desc    : 这里主要记录streamlit中的可视化方法


import numpy as np
import pandas as pd
import streamlit as st

st.title('Bar Chart - 条形图')
# Defining dataframe with its values
df = pd.DataFrame(
    np.random.randn(40, 4),
    columns=["C1", "C2", "C3", "C4"])
# Bar Chart
st.bar_chart(df)

st.title('Line Chart - 折线图')
# Defining dataframe with its values
df = pd.DataFrame(
    np.random.randn(40, 4),
    columns=["C1", "C2", "C3", "C4"])
# Bar Chart
st.line_chart(df)

st.title('Area Chart - 面积图')
# Defining dataframe with its values
df = pd.DataFrame(
    np.random.randn(40, 4),
    columns=["C1", "C2", "C3", "C4"])
# Bar Chart
st.area_chart(df)

st.title('Map - 地图')
# Defining Latitude and Longitude
locate_map = pd.DataFrame(
    np.random.randn(50, 2) / [10, 10] + [30.4589, 105.0078], columns=['latitude', 'longitude'])
# Map Function
st.map(locate_map)

st.title('Graphviz - 流程图')
# Creating graph object
st.graphviz_chart('''
digraph {
"Training Data" -> "ML Algorithm"
"ML Algorithm" -> "Model"
"Model" -> "Result Forecasting"
"New Data" -> "Model"
}
''')
# import graphviz as graphviz
# # Create a graphlib graph object
# graph = graphviz.Digraph()
# graph.edge('Training Data', 'ML Algorithm')
# graph.edge('ML Algorithm', 'Model')
# graph.edge('Model', 'Result Forecasting')
# graph.edge('New Data', 'Model')
# st.graphviz_chart(graph)

import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

st.title('Seaborn')
# Data Set
df = pd.read_csv("./files/avocado.csv")
# Defining Count Graph/Plot
fig = plt.figure(figsize=(10, 5))
sns.countplot(x="year", data=df)
st.pyplot(fig)

# Defining Violin Graph
fig = plt.figure(figsize=(10, 5))
sns.violinplot(x="year", y="AveragePrice", data=df)
st.pyplot(fig)

# Defining Strip Plot
fig = plt.figure(figsize=(10, 5))
sns.stripplot(x = "year", y="AveragePrice", data = df)
st.pyplot(fig)

