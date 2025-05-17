import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# NLTK resources download
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load models and vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")
rf_model = joblib.load("random_forest_model.pkl")
mnb_model = joblib.load("multinomial_naive_bayes_model.pkl")

# Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.write("Paste a news article below to detect whether it is **Genuine** or **Misleading**.")

news_input = st.text_area("Enter News Article:", height=300)

if st.button("Detect"):
    if not news_input.strip():
        st.warning("Please enter a news article.")
    else:
        # Preprocess and vectorize
        processed_text = preprocess_text(news_input)
        vectorized_input = vectorizer.transform([processed_text])

        # Predict using both models
        mnb_probs = mnb_model.predict_proba(vectorized_input)
        rf_probs = rf_model.predict_proba(vectorized_input)

        # Ensemble: soft voting
        avg_probs = (mnb_probs + rf_probs) / 2
        prediction = np.argmax(avg_probs, axis=1)[0]
        confidence = np.max(avg_probs) * 100

        # Output
        if prediction == 0:
            st.error(f"⚠️ This news seems **Misleading**. (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"✅ This news seems **Genuine**. (Confidence: {confidence:.2f}%)")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
