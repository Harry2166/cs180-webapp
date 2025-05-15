import streamlit as st
import pandas as pd
import numpy as np

# we should think of something witty here for the title
title = "king lebron"

st.set_page_config(
    page_title="cs180",
    page_icon="🌧️",
)

st.title(f"Welcome to :rainbow[{title}]")

st.write("## About the Project")
st.markdown(
    """
    Lorem ipsum dolor sit amet, **consectetur adipiscing elit**, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. 
    """
)

st.write("## Training Data")
# i'll place the data here later
st.write("Here is our training data:")
df = pd.DataFrame(np.random.randn(50, 20), columns=("col %d" % i for i in range(20)))
st.dataframe(df)
