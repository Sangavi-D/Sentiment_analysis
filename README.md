
# SENTIMENT ANALYSIS USING HUGGING FACE TRANSFORMER

A pretrained sentiment analysis model has been deployed in hugging face space with login details stored in  AWS RDS, making it accessible through a web application built with Streamlit.


## Requirements

transformers

torch

scipy

mysql-connector-python
## Steps involved

1) Install/Import necessary libraries.

2) A pretrained sentiment analysis model is found from hugging face models(https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) and used for sentiment analysis.

3) The app.py file contains the code for streamlit web application where the user can login and enter text to analyze the sentiment of the text.

4) The user details like username and login time is stored in AWS RDS.

5) The app.py file is stored in S3 bucket for future use.


## Getting started

1) Install/Import necessary libraries.Requirements.txt file is provided.

2) Create an AWS RDS instance to store user details and modify the host,user,port and password details in app.py file.

3) Create a space in hugging face and add the app.py and requirents.txt files and run the app to view the web application.

4) This the link to the streamlit web application running in hugging face space (https://huggingface.co/spaces/Sangavi16/sentiment_analysis)
## Skill takeaway
Transformers, Hugging face models, Streamlit, AWS RDS, AWS S3