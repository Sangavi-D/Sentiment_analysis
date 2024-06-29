from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import streamlit as st
from datetime import datetime
from scipy.special import softmax

st.title("SENTIMENT ANALYSIS USING HUGGING FACE TRANSFORMER")

import mysql.connector

connection = mysql.connector.connect(
  host = "Enter host",
  port = 4000,
  user = "Enter user",
  password = "Enter password",
  database = "sentimentanalysis"  
  
)
mycursor = connection.cursor(buffered=True)


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def analyze_text(text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    result = []
    for i in range(scores.shape[0]):
        label = config.id2label[ranking[i]]
        score = np.round(float(scores[ranking[i]]), 4)
        result.append(f"{i+1}) {label} {score}")
    return result


username = st.text_input("Username:", key="username")
login_successful = False  # Flag to track login status

if st.button("Login", key="login"):
    if username:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        sql = "INSERT INTO user_info(Username, Login_time) VALUES (%s, %s)"
        values = (username, current_time)
        mycursor.execute(sql, values)
        connection.commit()

        st.success(f"Login successful for {username} at {current_time}")
        login_successful = True  # Update flag on successful login

text_input = st.text_input("Enter text to analyze:")
analysis_result = None  # Initialize variable to store result

if login_successful:  # Check login status before analyzing text
    if text_input:
        analysis_result = analyze_text(text_input)

if analysis_result:
    st.subheader("Sentiment Analysis Result:")
    for item in analysis_result:
        st.write(item)
else:
    if login_successful:  # Only show this message if logged in
        st.warning("No text entered.")





     





  
  

 
  


  
 



