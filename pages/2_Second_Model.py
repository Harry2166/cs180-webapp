import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel 
import joblib
import os

st.set_page_config(page_title="Second Model", page_icon="👤")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RF_MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")

prediction_classes = {
    0: "risk",
    1: "neutral",
    2: "opportunity"
}

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
st.markdown("# Second Model")
st.write(
    """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    """
)

st.write("## Try Our Model")
user_input = st.text_input("Enter your climate-related text", "")
if user_input:
    embedding = get_embedding(user_input, tokenizer, model)
    prediction = rfc.predict([embedding])
    st.success(f"Predicted class: {prediction_classes[prediction[0]]}")
