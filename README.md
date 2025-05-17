
# 📰 Fake News Detection Web App

This is a **Streamlit-based web application** developed during my AI/ML internship to detect whether a news article is **Genuine** or **Misleading**. The app uses machine learning models trained on labeled news data and combines them using an ensemble approach for improved accuracy.

---

## 🚀 Features

- Accepts news text input from users
- Preprocesses the text using NLP techniques:
  - Lowercasing
  - Removing special characters
  - Stopword removal
  - Lemmatization
- Transforms text using **TF-IDF Vectorizer**
- Predicts using an ensemble of:
  - Multinomial Naive Bayes
  - Random Forest Classifier
- Displays clear output as **Genuine** or **Misleading**

---

## 🧠 Model Details

- The models were trained using scikit-learn.
- Soft voting (averaging probabilities) is used for final prediction.
- Models and vectorizer are saved using `joblib`.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python  
- **ML Models**: Scikit-learn (Naive Bayes, Random Forest)  
- **NLP**: NLTK (stopwords, lemmatization)  
- **Vectorization**: TF-IDF  
- **Model Serialization**: Joblib  

---

## 📂 Folder Structure

```
fake-news-app/
│
├── app.py                     # Streamlit Web App
├── tfidf_vectorizer.pkl       # Saved TF-IDF Vectorizer
├── random_forest_model.pkl    # Trained Random Forest Model
├── multinomial_naive_bayes_model.pkl  # Trained Naive Bayes Model
├── requirements.txt           # Required Python packages
└── README.md                  # Project Documentation
```

---

## 💻 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/MuhammadZahran26/Fake-News-Detection-App.git
cd fake-news-app
```

### 2. Install the dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

---

## ✅ Example Usage

Paste a paragraph or article in the input box and click "Detect".  
The app will analyze the text and tell you whether it's **Genuine** ✅ or **Misleading** ⚠️.

---

## 📌 Note

- Make sure you have the required `.pkl` files (models and vectorizer) in the same directory.
- For best results, use real news content (English only).

---

## 📬 Contact

For any issues or suggestions, feel free to open an issue or contact me via GitHub.

---


