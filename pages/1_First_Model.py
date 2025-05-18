import streamlit as st
import os
import torch
import joblib
from transformers import AutoTokenizer, AutoModel 
from layout import prediction_classes, section 

st.set_page_config(page_title="First Model", page_icon="👤")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RF_MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")
LR_MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_regression.pkl")

@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("Harry2166/fine-tuned-climate-bert")
    bert_model = AutoModel.from_pretrained("Harry2166/fine-tuned-climate-bert")
    ml_model = joblib.load(LR_MODEL_PATH)
    return tokenizer, bert_model, ml_model 

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

tokenizer, model, ml = load_models()

section("# BERT embeddings w/ RandomForestClassfiers & Logistic Regression", "")

section("## Methodology", 
    """
    BERT stands for Bidirectional Encoder Representations from Transformers and it is a deep neural language model trained and fine-tuned on unlabeled text to predict what sentences or words would follow one another accurately. 

Alongside this, in a paper conducted by [Krishnan and Anoop (2023)](https://arxiv.org/abs/2310.08099), running BERT, (specifically the ClimateBERT model) and feeding its embeddings (vector representations that capture the semantic meaning of the text) to several ML algorithms was done in order to classify the sentiment of multiple tweets about climate change. 

Among the ML algorithms used, Random Forest classification produced the best results in terms of accuracy. Logistic Regression was also tested due to its capability of balancing the weights of the classes, which is useful due to the imbalance in the dataset.

For this option, we are opting to use the “bert-base-uncased” BERT model and fine tuning based on the Medium article by [Amit (2025)](https://medium.com/@heyamit10/fine-tuning-bert-for-sentiment-analysis-a-practical-guide-f3d9c9cac236).
    """
)

section("### Process", 
    """
    Detailed below is the general process for how this methodology would be done:

1. Preprocess text (tokenizing, padding and truncation of text, sampling labels, etc.)
2. Use Optuna to find the best fine-tuning settings (e.g., learning rate, batch size).
3. Train BERT on labeled sentiment data using the best parameters from Optuna.
4. Get embeddings from the fine-tuned BERT model via mean pooling.
5. Use these embeddings to train a Random Forest Classifier and Logistic Regression model for final sentiment prediction.
6. Classify sentiment using the ML models.
7. Select which model performed the best on the development set.

    """
)

section("## Results")
st.write("Out of the two trained ML models, the model that produced the best results was the trained Logistic Regression model. Hence, this model was selected and can be tested below.")

section("## Try Our Model", "")
user_input = st.text_input("Enter your climate-related text", "")
if user_input:
    embedding = get_embedding(user_input, tokenizer, model)
    prediction = ml.predict([embedding])
    st.success(f"Predicted class: {prediction_classes[prediction[0]]}")
