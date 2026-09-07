# 🔎 Factify: Indian News Authenticity Detector

Factify is an NLP and Machine Learning application designed to screen news claims and headlines for stylistic and lexical misinformation patterns using a **Linear Support Vector Machine (LinearSVC)** and **TF-IDF Vectorization**.

The model is trained on the **Indian Fake News Dataset (IFND)**, incorporating verified claims from Indian fact-checking platforms such as AltNews, BoomLive, and PIB Fact Check.

---

## 🚀 Features
- **Fast Inference:** Lightweight Linear SVM trained on 10,000 unigram/bigram features.
- **Calibrated Probabilities:** Sigmoid transformation on hyperplane decision margin scores.
- **Interactive UI:** Clean Streamlit web interface with real-time prediction feedback.

---

## 🛠️ Project Structure
```text
Factify/
├── app.py                     # Streamlit frontend application
├── train.py                   # Data ingestion, vectorization & model training
├── requirements.txt           # Environment dependencies
├── svm_model.pkl              # Serialized SVM model
├── tfidf_vectorizer.pkl       # Serialized TF-IDF vectorizer
├── Factify_Presentation.pptx  # Presentation deck
└── README.md                  # Project documentation

⚙️ Installation & Setup
1. Clone the Repository:
Bash
git clone [https://github.com/omkar-vitb/Factify-Fake-News-Detection.git](https://github.com/YOUR_USERNAME/Factify-Fake-News-Detection.git)
cd Factify-Fake-News-Detection

2.
