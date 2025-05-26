# cs180-webapp
This is the designated web application for group 13: king lebron done for the project in fulfillment of the requirements for CS180.

## Models
The models used in this project are the following:
1. Logistic Regression and RandomForestClassfier models trained on custom fine-tuned BERT model's embeddings that was built on top of [distilbert](https://huggingface.co/distilbert/distilbert-base-uncased).
2. Birectional LSTM using embeddings from a Word2Vec model pre-trained on Google news articles.

You may access the trained models here:
1. [BERT and ML models](https://huggingface.co/Harry2166/fine-tuned-climate-bert/tree/main)
2. [Bidrectional LSTM](https://huggingface.co/cdvitug/bilstm)

## Prerequisites
1. It is recommended to use Google Colab as your platform in running the Python notebooks.
2. Ensure that you have access to the testing, training, and development data.
3. In HuggingFaceHub, create a token in your [settings tab](https://huggingface.co/settings/tokens).
4. Set the token as a secret in your Google Colab and name it as `HF_TOKEN`.

## Steps
### BERT model
0. In `bert-fine_tuning-ml-training.ipynb`, place the training data and development data into the files sidebar.
1. You may run the code right now if you want to fine-tune your own BERT model that is built on top of distilbert.
2. In `bert-ml-evaluation.ipynb`,  place the development data into the files sidebar.
3. You may run the code right to see the performance relative to the development set.
4. In running `demo_code.ipynb`, you may opt to run using the logistic regression model or the random forest classifier.

### BiLSTM model
0. In `bilstm-fine_tuning-ml-training.ipynb`, place the training data and development data into the files sidebar.
1. Run the code to find the best hyperparameters and build your own Bidirectional LSTM model.
2. In `bilstm-ml-evaluation.ipynb`,  place the development data into the files sidebar.
3. You may run the code right to see the performance relative to the development set.
4. In running `demo_code.ipynb`, you may run the best model found in our runs to test an uploaded csv of texts.

## Team

The following are the developers of the project:
1. [Nathaniel Feliciano](https://github.com/natecomsci)
2. [Prince Harry Quijano](https://github.com/Harry2166)
3. [Carl Geevee Vitug](https://github.com/good-vibe)
