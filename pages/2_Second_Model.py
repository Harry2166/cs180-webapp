import streamlit as st
from layout import prediction_classes, section 
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download, login
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

REPO = "cdvitug/bilstm"
FILENAME = "bidirectional_lstm_model.h5"
TOKENIZER_MODEL_PATH = hf_hub_download(repo_id=REPO, filename="tokenizer.json")

with open(TOKENIZER_MODEL_PATH, "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

tokenizer = tokenizer_from_json(tokenizer_data) 

hf_token = st.secrets["HF_TOKEN"]
login(token=hf_token)
st.set_page_config(page_title="Bidirectional LSTM", page_icon="👤")

section("# Second Model", "")
section("## Methodology", """
This approach will tackle text sentiment analysis through a recurrent neural network, specifically through a bidirectional LSTM or BiLSTM. BiLSTMs allow meaningful relationships and contextual information to be sent both forwards and backwards in a provided text. 

A set of vectors are used which contains 300-dimensional vectors for 3 million words and phrases. These come from a pre-trained Word2Vec model that was trained on Google News articles. Using the vectors for each word to create an embedding matrix, the model contains an embedding layer, 2 bidirectional LSTM layers and finally the output dense layer for 3 classifications.

Keras Tuner was also used to find the best parameters at each part of the layers to get the best results.
""")

section("### Process", """
Detailed below is the general process for how the model would be done:
1. Prepare Word2Vec vectors from ‘word2vec-google-news-300’
2. Text will be tokenized, converted to sequences and padded.
3. Model compiled with an embedding layer, 2 BiLSTM layers and the output layer.
4. Best hyperparameters found
5. Model refitted to hyperparameters
6. Model predicts.
""")
section("## Results", "")
section("## Try Our Model", "")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
model_path = hf_hub_download(repo_id=REPO, filename=FILENAME)
ml = load_model(model_path)
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    text_column = st.selectbox("Select the column containing the climate-related text", df.columns)

    if st.button("Run Classification"):
        try:

            token_text = tokenizer.texts_to_sequences(df[text_column].astype(str))
            token_text = pad_sequences(token_text, maxlen=100)

            X = np.vstack(token_text)
            predictions = ml.predict(X)
            predicted_classes = np.argmax(predictions, axis=1)

            df["Prediction"] = [p for p in predicted_classes]
            df["Converted Prediction"] = [prediction_classes[p] for p in predicted_classes]

            st.success("Classification complete!")
            st.write(df)

        except Exception as e:
            st.error(f"Error: {e}")

