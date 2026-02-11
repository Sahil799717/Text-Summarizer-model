import streamlit as st
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Text Summarizer Tool", layout="wide")

st.title("📝 Text Summarizer Tool")
st.write("Generate a concise summary from long text using NLP.")

# Download required tokenizer (only first time)
nltk.download("punkt")

# -----------------------------
# Summarization Function
# -----------------------------
def summarize_text(text, num_sentences=3):
    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Convert matrix to flat array of scores
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    # Get indices of top sentences
    ranked_indices = np.argsort(sentence_scores)[::-1]

    # Select top N sentences
    selected_indices = sorted(ranked_indices[:num_sentences])

    summary = " ".join([sentences[i] for i in selected_indices])

    return summary

# -----------------------------
# User Input
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
