import streamlit as st
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import textwrap

st.set_page_config(page_title="Mini RAG (TF-IDF)", layout="centered")
st.title("Mini RAG (TF-IDF Retriever)")

MODEL = joblib.load(Path(__file__).parent / "tfidf.joblib")
vec = MODEL["vectorizer"]
X = MODEL["matrix"]
names = MODEL["names"]
texts = MODEL["texts"]

q = st.text_input("Ask a question about your docs")
k = st.slider("Top-K", 1, 5, 3)
if st.button("Search") and q.strip():
    qv = vec.transform([q])
    sims = cosine_similarity(qv, X).ravel()
    top_idx = np.argsort(sims)[::-1][:k]
    st.subheader("Top Passages")
    for i, idx in enumerate(top_idx, 1):
        st.markdown(f"**{i}. {names[idx]}** (score={sims[idx]:.3f})")
        snippet = textwrap.shorten(texts[idx].replace("\n"," "), width=300, placeholder=" ...")
        st.write(snippet)

    best = top_idx[0]
    st.subheader("Concise Answer (extractive)")
    st.write(textwrap.shorten(texts[best].replace("\n"," "), width=500, placeholder=" ..."))
