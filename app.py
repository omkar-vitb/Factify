import pickle
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Factify-Fake News Detector",
    page_icon="🔎",
    layout="centered"
)
@st.cache_resource
def load_artifacts():
    with open("svm_model.pkl","rb") as f:
        model=pickle.load(f)
    with open("tfidf_vectorizer.pkl","rb") as f:
        vectorizer=pickle.load(f)
    return model,vectorizer

model, vectorizer=load_artifacts()

st.title=(("🔎 Factify"))
st.subheader("Indian News Authenticity detector (SVM)")
st.write("Enter a news headline or claim to analyze its language patterns")

user_text=st.text_area(
    "News Headline/Statement",
    placeholder="e.g, Paste headline here...",
    height=120
)

if st.button("Check Authencity", use_container_width=True):
    cleaned=user_text.strip()
    if not cleaned:
        st.warning("Please enter some text first.")
    else:
        vec = vectorizer.transform([cleaned])
        score = model.decision_function(vec)[0]

        prob_real=1/(1+np.exp(-score))
        st.divider()

        if abs(score) < 0.4:
            st.warning("⚠️ **Inconclusive / Low Certainty**")
            st.write(
                "The model cannot confidently verify this claim based on writing patterns alone. "
                "Manual fact-checking is recommended."
            )
            st.caption(f"Score Margin: {score:.2f} | Estimated Probability: {prob_real * 100:.1f}%")
        elif score > 0:
            st.success("✅ **Result: Likely Real News**")
            st.metric("Confidence", f"{prob_real * 100:.1f}%")
        else:
            st.error("🚨 **Result: Likely Fake News**")
            st.metric("Confidence", f"{(1 - prob_real) * 100:.1f}%")