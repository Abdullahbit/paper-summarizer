# modules/tfidf_ranker.py (replace rank_sentences with below)
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def textrank_sentences(sentences, top_n=5):
    

    if len(sentences) <= top_n:
        return sentences
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    sim_mat = cosine_similarity(X)
    # build graph
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    # sort by score, take top_n, then sort those by original order
    ranked_idx = sorted(scores, key=scores.get, reverse=True)[:top_n]
    ranked_idx = sorted(ranked_idx)  # preserve original order for readability
    return [sentences[i] for i in ranked_idx]

# You can keep the old TF-IDF ranker as an auxiliary method if you want to compare.
