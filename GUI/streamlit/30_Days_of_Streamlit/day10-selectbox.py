'''https://30days.streamlit.app/?challenge=Day+10'''


import streamlit as st

st.header('st.selectbox')

option = st.selectbox(
    'What is your favorite color?',
    ('Blue', 'Red', 'Green'))

st.write('Your favorite color is ', option)
