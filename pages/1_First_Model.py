import streamlit as st
from layout import prediction_classes, section 

st.set_page_config(page_title="First Model", page_icon="👤")

section("# BERT embeddings w/ Random Forest Classifier", "")

section("## Methodology", ""
    """
    BERT stands for Bidirectional Encoder Representations from Transformers and it is a deep neural language model trained and fine-tuned on unlabeled text to predict what sentences or words would follow one another accurately. 

Alongside this, in a paper conducted by [Krishnan and Anoop (2023)](https://arxiv.org/abs/2310.08099), running BERT, (specifically the ClimateBERT model) and feeding its embeddings (vector representations that capture the semantic meaning of the text) to several ML algorithms was done in order to classify the sentiment of multiple tweets about climate change. Among the ML algorithms used, Random Forest classification produced the best results in terms of accuracy. Hence, we will be using random forest classification for this process.

For this option, we are opting to use the “bert-base-uncased” BERT model and fine tuning based on the Medium article by [Amit (2025)](https://medium.com/@heyamit10/fine-tuning-bert-for-sentiment-analysis-a-practical-guide-f3d9c9cac236).

    """
)

section("### Process", ""
    """
    Detailed below is the general process for how this methodology would be done:

1. Preprocess text (tokenizing, padding and truncation of text, sampling labels, etc.)
2. Use Optuna to find the best fine-tuning settings (e.g., learning rate, batch size).
3. Train BERT on labeled sentiment data using the best parameters from Optuna.
4. Get embeddings from the fine-tuned BERT model via mean pooling.
5. Use these embeddings to train a Random Forest Classifier for final sentiment prediction.
6. Classify sentiment using the trained RFC.


    """
)


section("## Results")
section("## Try Our Model", "")
