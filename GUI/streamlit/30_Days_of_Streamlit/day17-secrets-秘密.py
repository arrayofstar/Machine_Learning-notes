#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-03-31 15:18
# @Author  : Dreamstar
# @File    : day17-secrets-秘密.py
# @Desc    : https://30days.streamlit.app/?challenge=Day+17
# @Desc    : st.secrets允许您存储机密信息，如API密钥、数据库密码或其他凭据。
# @Desc    : 在本地运行时，数据会储存在.streamlit/secrets.toml文件之下，所以这个密码的保存方式看上不并不是特别的好。


import streamlit as st

st.title('st.secrets')

st.write(st.secrets['message'])