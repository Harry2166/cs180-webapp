import streamlit as st
from layout import prediction_classes, section 

st.set_page_config(page_title="Second Model", page_icon="👤")

section("# Second Model")
section("## Methodology")
section("## Results")
section("## Try Our Model", "")

user_input = st.text_input("Enter your climate-related text", "")
if user_input:
    st.success(f"Predicted class: ???") 
