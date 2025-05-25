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
import os

REPO = "cdvitug/bilstm"
FILENAME = "bidirectional_lstm_model.h5"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOKENIZER_MODEL_PATH = os.path.join(BASE_DIR, "models", "tokenizer.json")

with open(TOKENIZER_MODEL_PATH, "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

tokenizer = tokenizer_from_json(tokenizer_data) 

hf_token = st.secrets["HF_TOKEN"]
login(token=hf_token)
st.set_page_config(page_title="Second Model", page_icon="👤")

section("# Second Model")
section("## Methodology")
section("## Results")
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

            df["Prediction"] = [prediction_classes[p] for p in predictions]

            st.success("Classification complete!")
            st.write(df)

        except Exception as e:
            st.error(f"Error: {e}")

