# -*- coding: utf-8 -*-
# @Time    : 2023/6/11 11:04
# @Author  : Dreamstar
# @File    : day22-form-表单.py
# @Link    : https://30days.streamlit.app/?challenge=Day22
# @Desc    : 表单是一个视觉上将元素和组件编组的容器，包含一个提交按钮，只有当提交的时候，才会出发交互操作。


import streamlit as st

st.title('st.form')

# Full example of using the with notation
st.header('1. Example of using `with` notation')
st.subheader('Coffee machine')

with st.form('my_form'):
    st.subheader('**Order your coffee**')

    # Input widgets
    coffee_bean_val = st.selectbox('Coffee bean', ['Arabica', 'Robusta'])
    coffee_roast_val = st.selectbox('Coffee roast', ['Light', 'Medium', 'Dark'])
    brewing_val = st.selectbox('Brewing method', ['Aeropress', 'Drip', 'French press', 'Moka pot', 'Siphon'])
    serving_type_val = st.selectbox('Serving format', ['Hot', 'Iced', 'Frappe'])
    milk_val = st.select_slider('Milk intensity', ['None', 'Low', 'Medium', 'High'])
    owncup_val = st.checkbox('Bring own cup')

    # Every form must have a submit button
    submitted = st.form_submit_button('Submit')

if submitted:
    st.markdown(f'''
        ☕ You have ordered:
        - Coffee bean: `{coffee_bean_val}`
        - Coffee roast: `{coffee_roast_val}`
        - Brewing: `{brewing_val}`
        - Serving type: `{serving_type_val}`
        - Milk: `{milk_val}`
        - Bring own cup: `{owncup_val}`
        ''')
else:
    st.write('☝️ Place your order!')


# Short example of using an object notation
st.header('2. Example of object notation')

form = st.form('my_form_2')
selected_val = form.slider('Select a value')
form.form_submit_button('Submit')

st.write('Selected value: ', selected_val)