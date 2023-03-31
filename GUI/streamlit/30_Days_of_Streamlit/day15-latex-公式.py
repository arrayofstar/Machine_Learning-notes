#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-03-31 15:01
# @Author  : Dreamstar
# @File    : day15-latex-公式.py
# @Desc    : https://30days.streamlit.app/?challenge=Day+15

import streamlit as st

st.header('st.latex')

st.latex(r'''
     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
     \sum_{k=0}^{n-1} ar^k =
     a \left(\frac{1-r^{n}}{1-r}\right)
     ''')