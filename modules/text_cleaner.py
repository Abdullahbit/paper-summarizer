# modules/text_cleaner.py (replace or extend)
import re
import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

def extract_main_content(text: str) -> str:
    """
    Try to start from the Abstract (or Introduction) to skip author lists, headers, etc.
    """
    # Normalize line breaks
    text = text.replace("\r", " ").replace("\n", " ")
    # Find "Abstract" or "Introduction" to start the main content
    match = re.search(r"\babstract\b", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start():]
    else:
        match = re.search(r"\bintroduction\b", text, flags=re.IGNORECASE)
        if match:
            text = text[match.start():]
    return text

def remove_artifacts(text: str) -> str:
    """
    Remove citations, URLs, figure/table labels, numbers, and redundant spaces.
    """
    # Citations like [1], [1,2], [12–14]
    text = re.sub(r"\[[0-9,\-\s]+\]", " ", text)
    # Citations like (Smith et al., 2020)
    text = re.sub(r"\([A-Z][a-z]+ et al\.,?\s*\d{4}\)", " ", text)
    # URLs and hyperlinks
    text = re.sub(r"https?://\S+", " ", text)
    # Remove figure/table labels
    text = re.sub(r"(Figure|Table)\s*\d+[A-Za-z\-]*", " ", text)
    # Remove leftover percentage/stat lines (e.g., 0% 20% 40%)
    text = re.sub(r"(\d+%)+", " ", text)
    # Remove weird sequences of dots, dashes, etc.
    text = re.sub(r"[\.\-]{3,}", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_and_split(text: str):
    """
    Clean the raw PDF text, remove artifacts, and split it into meaningful sentences.
    """
    text = extract_main_content(text)
    text = remove_artifacts(text)

    # Sentence segmentation
    sentences = nltk.sent_tokenize(text)

    # Filter out too-short or too-long sentences
    cleaned = []
    for s in sentences:
        s = s.strip()
        if len(s.split()) <= 5:  # very short
            continue
        if len(s) > 400:         # suspiciously long (likely table text)
            continue
        # Skip author-like lines (many commas + many capitalized words)
        if s.count(",") >= 4 and sum(1 for w in s.split() if w[:1].isupper()) > 6:
            continue
        cleaned.append(s)

    return cleaned