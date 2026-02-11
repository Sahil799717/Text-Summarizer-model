import streamlit as st
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Text Summarizer Tool", layout="wide")

st.title("📝 Text Summarizer Tool")
st.write("Generate a concise summary from long text using NLP (TF-IDF based).")

# -----------------------------
# Sentence Splitter (No NLTK)
# -----------------------------
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return [s for s in sentences if len(s) > 20]

# -----------------------------
# Summarization Function
# -----------------------------
def summarize_text(text, num_sentences=3):
    sentences = split_sentences(text)

    if len(sentences) == 0:
        return "Not enough content to summarize."

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    ranked_indices = np.argsort(sentence_scores)[::-1]

    selected_indices = sorted(ranked_indices[:num_sentences])

    summary = " ".join([sentences[i] for i in selected_indices])

    return summary

# -----------------------------
# UI
# -----------------------------
text_input = st.text_area("Enter your text here:", height=250)

num_sentences = st.slider("Select number of summary sentences:", 1, 10, 3)

if st.button("Generate Summary"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        summary = summarize_text(text_input, num_sentences)
        st.subheader("📌 Summary:")
        st.success(summary)
