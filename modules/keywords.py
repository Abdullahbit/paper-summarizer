# modules/keywords.py tweak
from modules.text_cleaner import extract_main_content, remove_artifacts
from modules.tfidf_ranker import textrank_sentences as textrank_sentences


def top_keywords(text, top_n=10):
    from sklearn.feature_extraction.text import TfidfVectorizer
    text = remove_artifacts(text)
    text = extract_main_content(text)
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    X = vectorizer.fit_transform([text])
    scores = X.toarray()[0]
    words = vectorizer.get_feature_names_out()
    top_idx = scores.argsort()[-top_n:][::-1]
    return [words[i] for i in top_idx]
