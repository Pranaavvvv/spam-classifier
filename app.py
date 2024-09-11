import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import re

dataset = pd.read_csv('spam.csv')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

dataset['Message'] = dataset['Message'].apply(preprocess_text)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dataset['Message'])
y = dataset['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://plus.unsplash.com/premium_photo-1685287731741-3c21312e4088?q=80&w=1776&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .main {
        background-color: rgba(5, 5, 5, 0.8);  
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .title {
        font-family: 'Arial', sans-serif;
        color: #ffffff;
        text-align: center;
        margin-bottom: 20px;
    }
    .input-area {
        background-color: #f8f9fc;
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .btn {
        background-color: #0066cc;
        color: #ffffff;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: 0.3s;
    }
    .btn:hover {
        background-color: #004c99;
    }
    .result {
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        color: #ffffff;
        padding: 10px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .result-spam {
        background-color: #d9534f;
    }
    .result-ham {
        background-color: #5cb85c;
    }
    footer {
        text-align: center;
        padding: 20px;
        font-size: 14px;
        color: #888888;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>📩 Spam Classifier</h1>", unsafe_allow_html=True)
st.write("🔍 Enter a message below to classify it as **Spam** or **Ham**.")

col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://img.generation-nt.com/0001661204.jpg", width=150)  

with col2:
    user_message = st.text_area("Message", height=150, placeholder="Type your message here...", 
                                help="Enter the message you want to classify")

if st.button('📬 Classify Message', help="Click to classify the message"):
    if user_message:
        user_message = preprocess_text(user_message)
        message_vector = vectorizer.transform([user_message])
        prediction = model.predict(message_vector)
        result_class = 'result-spam' if prediction[0] == 'spam' else 'result-ham'
        result_text = '🚫 Spam' if prediction[0] == 'spam' else '✅ Ham'
        st.markdown(f"<div class='result {result_class}'>{result_text}</div>", unsafe_allow_html=True)
    else:
        st.error('⚠️ Please enter a message to classify.', icon="⚠️")

st.markdown(
    """
    <footer>
        &copy; 2024 Spam Classifier | Built with ❤ 
    </footer>
    """,
    unsafe_allow_html=True
)
