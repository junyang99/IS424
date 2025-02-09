import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
with open("multinomialnb.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the vectorizer
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Streamlit App UI
st.title("Fake News Detector")

# User Input
news_title = st.text_input("Enter News Title:")
news_text = st.text_area("Enter News Content:")

# Prediction Function
def predict_fake_news(title, text):
    full_text = title + " " + text  # Combine title and content
    text_vectorized = vectorizer.transform([full_text])
    prediction = model.predict(text_vectorized)[0]
    return "True News" if prediction == 1 else "Fake News"

# Predict Button
if st.button("Check Authenticity"):
    if news_title and news_text:
        result = predict_fake_news(news_title, news_text)
        st.subheader(f"Prediction: {result}")
    else:
        st.warning("Please enter both title and content of the news.")
