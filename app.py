import streamlit as st
from modules.pdf_reader import extract_text
from modules.text_cleaner import clean_and_split
from modules.tfidf_ranker import textrank_sentences
from modules.keywords import top_keywords
from modules.ai_summary import ai_summarize

st.title("📄 Research Paper Summarizer")

uploaded = st.file_uploader("Upload a PDF", type="pdf")
if uploaded:
    text = extract_text(uploaded)
    sentences = clean_and_split(text)
        # Stage 1: Extractive summary 

    extractive_summary = " ".join(textrank_sentences(sentences)) 
    summary = textrank_sentences(sentences)
    keywords = top_keywords(text)
    st.subheader("Summary")
    st.write(" ".join(summary))
    st.subheader("Keywords")
    st.write(", ".join(keywords))
