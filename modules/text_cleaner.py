# 1. Refactored modules/text_cleaner.py
import re
import nltk
from nltk.tokenize import sent_tokenize
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

def clean_and_split(raw_text: str) -> list[str]:
    """
    Cleans raw text while preserving line breaks for better artifact detection,
    then performs targeted sentence splitting and filtering.
    """
    # 1. Preserve line breaks initially for better artifact isolation
    text = raw_text.replace("\r", "")
    
    # 2. Section Extraction (Find 'Introduction' or 'The Current Situation')
    # Use the original structure for better segmentation.
    # We use re.DOTALL to match across newlines.
    match = re.search(r"(INTRODUCTION|THE CURRENT SITUATION)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = text[match.start():]
    
    # 3. Targeted Cleaning - ONLY remove page artifacts using a small window
    # The heavy-handed ALL-CAPS and title removal is too destructive.
    
    # Remove "Wirth & Perkins - Learning to Learn" footer/header (based on the paper)
    text = re.sub(r"Wirth\s*&\s*Perkins\s*-\s*Learning\s*to\s*Learn", "\n", text)
    
    # Remove single-line page/section numbers
    text = re.sub(r"^\s*(\d+|\.|\s*-\s*|Page\s*\d+)\s*$", "\n", text, flags=re.MULTILINE)
    
    # Remove URLs (but keep surrounding text)
    text = re.sub(r"https?://\S+", " ", text)
    
    # Replace the broken list items (like `1)`) with proper periods to help NLTK
    text = re.sub(r"(\s*\d+\))\s*", ". ", text) # Replaces 1) with . 
    
    # 4. Normalize Whitespace and Split into Thematic Chunks
    # Replace multiple newlines with a unique delimiter to denote a section break
    text = re.sub(r"[\n]{2,}", " [SECTION_DELIMITER] ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # 5. Sentence Segmentation and Filtering
    
    # NLTK's sent_tokenize is good, but we can also use custom tokenizers if needed.
    sentences = nltk.sent_tokenize(text)
# Filter out table-like run-on sentences identified in your latest output
    cleaned = []
    for s in sentences:
        s = s.strip()
        # Your existing length filters are good:
        if len(s.split()) <= 5:   
            continue
        if len(s) > 400:          # Suspect run-on sentence
            continue
        
        # NEW FILTER: Target the specific run-on text from the Fink's table
        # We look for a high ratio of Title-Cased words (like those used as table headers)
        title_cased_words = sum(1 for w in s.split() if w[0].isupper() and w not in ['The', 'A', 'An'])
        if title_cased_words / len(s.split()) > 0.45: # If almost half the words are Title Case
            continue # Skip this sentence, it's likely part of a column/list run-on.

        # Skip author-like lines (many commas + many capitalized words) - KEEP THIS
        if s.count(",") >= 4 and sum(1 for w in s.split() if w[:1].isupper()) > 6:
            continue
        
        cleaned.append(s)
        
    return cleaned
# NOTE: The TFIDF-based ranker needs NO CHANGE, as its role is only to score.