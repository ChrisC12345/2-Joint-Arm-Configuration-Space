import streamlit as st
import random
import time

st.title("Test")
st.metric("Temperature", random.randint(60, 100))

time.sleep(1)
st.rerun()