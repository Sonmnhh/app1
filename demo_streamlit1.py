#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk


# In[2]:


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[6]:


# Load the model
model_path = 'models/NaiveBayes_model.sav'
try:
    with open(model_path, 'rb') as file:
        classifier = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}")

# Load the vectorizer
vectorizer_path = 'models/CountVectorizer.sav'
try:
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
except FileNotFoundError:
    st.error(f"Vectorizer file not found: {vectorizer_path}")

# Text preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# In[7]:


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\n',' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), "", text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


# In[8]:


# Streamlit interface
st.title('Financial Sentiment Analysis')

input_text = st.text_area("Enter text for sentiment analysis")

if st.button("Analyze"):
    if input_text:
        preprocessed_text = preprocess_text(input_text)
        text_vectorized = vectorizer.transform([preprocessed_text])
        prediction = classifier.predict(text_vectorized)
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        st.write(f"Predicted Sentiment: {sentiment_map[prediction[0]]}")
    else:
        st.write("Please enter some text for analysis.")


# In[ ]:




