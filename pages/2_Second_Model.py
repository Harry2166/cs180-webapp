import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel 
import joblib
import os
from layout import prediction_classes, section 

st.set_page_config(page_title="Second Model", page_icon="👤")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RF_MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")

@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("Harry2166/fine-tuned-climate-bert")
    bert_model = AutoModel.from_pretrained("Harry2166/fine-tuned-climate-bert")
    rf_model = joblib.load(RF_MODEL_PATH)
    return tokenizer, bert_model, rf_model

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

tokenizer, model, rfc = load_models()

section("# Second Model")
section("## Methodology")
section("## Results")
section("## Try Our Model", "")

user_input = st.text_input("Enter your climate-related text", "")
if user_input:
    embedding = get_embedding(user_input, tokenizer, model)
    prediction = rfc.predict([embedding])
    st.success(f"Predicted class: {prediction_classes[prediction[0]]}")
