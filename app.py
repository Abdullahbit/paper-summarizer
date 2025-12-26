import streamlit as st
from modules.pdf_reader import extract_text
from modules.text_cleaner import clean_and_split
from modules.tfidf_ranker import textrank_sentences
from modules.ai_summarizer import refine_with_openai
from modules.thematic_grouping import group_sentences_by_theme

st.set_page_config(page_title="Paper Summarizer", layout="wide")

st.title("📄 Research Paper Summarizer")
st.write("Upload a PDF to generate an extractive and optional AI-refined abstractive summary.")

# File Upload
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        try:
            raw_text = extract_text(uploaded_file)
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            st.stop()

    st.success("Text extracted successfully!")

    with st.spinner("Cleaning and splitting text..."):
        try:
            sentences = clean_and_split(raw_text)
        except Exception as e:
            st.error(f"Error cleaning text: {e}")
            st.stop()

    if len(sentences) == 0:
        st.error("No usable text found in the PDF after cleaning.")
        st.stop()

    # Sidebar Controls
    st.sidebar.header("Settings")

    top_n = st.sidebar.slider(
        "Number of sentences for extractive summary",
        min_value=2,
        max_value=10,
        value=5
    )

    st.subheader("Extractive Summary")
    with st.spinner("Generating extractive summary..."):
        try:
            extractive_list = textrank_sentences(sentences, top_n=top_n)
            extractive_summary = " ".join(extractive_list)
        except Exception as e:
            st.error(f"Error ranking sentences: {e}")
            st.stop()

    st.write(extractive_summary)


  # AI Refinement Section
    st.markdown("---")
    st.subheader("AI-Refined Summary (Abstractive)")

    refine_button = st.button("✨ Refine summary with OpenAI")

    if refine_button:
        with st.spinner("Refining summary using OpenAI..."):
            themed_input = group_sentences_by_theme(extractive_list) 
            
            # --- DIAGNOSTIC LINE ---
            st.subheader("Diagnostic: Thematic Input Sent to AI")
            st.json(themed_input) # Use st.json to display the dictionary clearly
            # -----------------------
            
            try:
                refined = refine_with_openai(themed_input) 
                st.write(refined)
            except Exception as e:
                st.error(f"AI refinement failed: {e}")
                st.info("Showing extractive summary as fallback.")
                st.write(extractive_summary)

    st.markdown("---")
    
    # Keywords (very basic TF-IDF-based keywords)
    st.subheader("Keywords (from extractive summary)")

    def extract_keywords(text, top_k=10):
        words = [w.lower() for w in text.split() if w.isalpha()]
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:top_k]]

    st.write(", ".join(extract_keywords(extractive_summary)))
