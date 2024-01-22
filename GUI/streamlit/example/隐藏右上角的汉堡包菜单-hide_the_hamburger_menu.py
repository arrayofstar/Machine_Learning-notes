# -*- coding: utf-8 -*-
# @Time    : 2023/6/22 0:54
# @Author  : Dreamstar
# @File    : 隐藏右上角的汉堡包菜单-hide_the_hamburger_menu.py
# @Link    : https://docs.streamlit.io/knowledge-base/using-streamlit/how-hide-hamburger-menu-app
# @Desc    : 可以使用附加CSS
# Streamlit allows developers to configure their hamburger menu to be more user-centric
# via st.set_page_config(). While you can configure menu items with st.set_page_config(),
# there is no native support to hide/remove the menu from your app.
# however, use an unofficial CSS hack with st.markdown() to hide the menu from your app.
# To do so, include the following code snippet in your app:


import streamlit as st

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

"# 让右上角的汉堡包菜单消失"