import os
import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))  # the saved pipeline references text_utils.clean_text
MODEL_PATH = os.path.join(ROOT, "models", "sentiment_pipeline.pkl")


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")
st.title("Customer Review Sentiment Analysis")
st.write("Enter a customer review to predict whether it is Positive, Negative or Neutral.")

pipeline = load_pipeline()

review = st.text_area("Review text", height=150, placeholder="e.g. The battery drains quickly and does not last.")

if st.button("Analyze", type="primary"):
    if not review.strip():
        st.warning("Please enter some review text first.")
    else:
        label = pipeline.predict([review])[0]
        probs = pd.Series(pipeline.predict_proba([review])[0], index=pipeline.classes_)
        st.subheader(f"Predicted sentiment: {label}")
        st.metric("Confidence", f"{probs[label]:.1%}")
        st.bar_chart(probs.sort_values(ascending=False))
        st.caption("Probabilities come from the saved model (models/sentiment_pipeline.pkl).")
