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
    In this project, our group participated in a shared task modeled after real-world challenges in the natural language processing
    (NLP) community. Shared tasks are collaborative benchmarks designed to evaluate and compare the performance of different models
    on specific NLP problems. Our particular task focused on text classification in the context of climate change, 
    aligning with broader efforts to leverage AI for social and environmental impact.

    We worked with the Climate Sentiment dataset—an expert-annotated collection of corporate disclosure 
    paragraphs—tasked with classifying each paragraph's sentiment regarding climate-related topics. 
    To address this challenge, we implemented and evaluated deep learning models, 
    including a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model 
    and LSTMs. 

    Our approach follows the structure of a closed shared
    task, using only the provided dataset to train and develop our models, with the goal of achieving high performance on sentiment
    classification and contributing insights into how NLP can support climate-related decision-making.
    """
)

# st.write("## Training Data")
# # i'll place the data here later
# st.write("Here is our training data:")
# df = pd.DataFrame(np.random.randn(50, 20), columns=("col %d" % i for i in range(20)))
# st.dataframe(df)
