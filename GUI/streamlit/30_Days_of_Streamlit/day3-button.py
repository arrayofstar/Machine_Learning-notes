'''https://30days.streamlit.app/?challenge=Day+3'''
import streamlit as st

st.header('st.button')

if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')
