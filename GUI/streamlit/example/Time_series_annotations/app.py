# -*- coding: utf-8 -*-
# @Time    : 2023/6/15 16:42
# @Author  : Dreamstar
# @File    : app.py.py
# @Link    : https://example-time-series-annotation.streamlit.app/
# @Desc    : 模仿案例

import altair as alt
import pandas as pd
import streamlit as st

@st.cache_data
def get_data():
    df_source = pd.read_csv('data/WTH_small.csv')
    source = df_source.iloc[:500]
    return source

@st.experimental_memo(ttl=60 * 60 * 24)
def get_chart(data):
    hover = alt.selection_single(
        fields=["date"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, height=500, title="Evolution of stock prices")
        .mark_line()
        .encode(
            x=alt.X("date", title="Date"),
            y=alt.Y("DewPointFarenheit", title="DewPointFarenheit"),
            # color="symbol",
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="yearmonthdate(date)",
            y="DewPointFarenheit",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("date", title="Date"),
                alt.Tooltip("DewPointFarenheit", title="DewPointFarenheit"),
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()

data = get_data()
chart = get_chart(data)

st.markdown("# 时间序列注释\n"
            "使用注释为您的时间序列提供更多上下文！")

col1, col2, col3 = st.columns(3)

col1.text_input("Choose a ticker (⬇💬👇ℹ️ ...)", value='⬇')
col2.slider("Horizontal offset", -30, 30, -1)
col3.slider("Vertical offset", -30, 30, -10)

# Input annotations
ANNOTATIONS = [
    ("01 01, 2010", "Pretty good day for GOOG"),
    ("01 02, 2010", "Something's going wrong for GOOG & AAPL"),
    ("01 03, 2010", "Market starts again thanks to..."),
    ("01 04, 2010", "Small crash for GOOG after..."),
]

# Create a chart with annotations
annotations_df = pd.DataFrame(ANNOTATIONS, columns=["date", "event"])
annotations_df.date = pd.to_datetime(annotations_df.date)
annotations_df["y"] = 0
annotation_layer = (
    alt.Chart(annotations_df)
    .mark_text(size=15, text="⬇", dx=-1, dy=-10, align="center")
    .encode(
        x="date:T",
        y=alt.Y("y:Q"),
        tooltip=["event"],
    )
    .interactive()
)

# Display both charts together
st.altair_chart((chart + annotation_layer).interactive(), use_container_width=True)