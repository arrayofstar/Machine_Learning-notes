import time

import streamlit as st

st.title('Spinner')

# Defining Spinner
with st.spinner('Loading...'):
    time.sleep(5)
st.write('Hello Data Scientists')