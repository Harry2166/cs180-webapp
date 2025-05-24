import streamlit as st
from layout import prediction_classes, section 
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download, login
from tensorflow.keras.models import load_model

REPO = "cdvitug/bilstm"
FILENAME = "bidirectional_lstm_model.h5"
st.set_page_config(page_title="Second Model", page_icon="👤")

section("# Second Model")
section("## Methodology")
section("## Results")
section("## Try Our Model", "")

hf_token = st.secrets["HF_TOKEN"]
login(token=hf_token)

uploaded_file = st.file_uploader("Upload CSV", type="csv")
model_path = hf_hub_download(repo_id=REPO, filename=FILENAME)
ml = load_model(model_path)
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    # Let user choose which column to use
    text_column = st.selectbox("Select the column containing the climate-related text", df.columns)

    if st.button("Run Classification"):
        try:
            # Convert to array and predict
            X = np.vstack(df[text_column])
            predictions = ml.predict(X)

            # Add results to dataframe
            df["Prediction"] = [prediction_classes[p] for p in predictions]

            st.success("Classification complete!")
            st.write(df)

            # Download link
            # csv = df.to_csv(index=False).encode('utf-8')
            # st.download_button("Download Results as CSV", csv, "classified_results.csv", "text/csv")

        except Exception as e:
            st.error(f"Error: {e}")

