# modules/thematic_grouping.py

from typing import List, Dict 

def group_sentences_by_theme(sentences: List[str]) -> Dict: # <-- Use List and Dict
    """
    Groups the selected extractive sentences into thematic clusters for
    feeding a coherent abstractive prompt to the Gemini API.
    """
    # 1. Define Thematic Keywords based on the paper's structure and core ideas
    THEMATIC_KEYWORDS = {
        "Foundational Shift": ["calls for new kinds of learning", "purpose of learning has recently shifted", "college education", "successful student must also know"],
        "Taxonomies and Development": ["Bloom’s taxonomy", "cognitive domain", "Krathwohl", "intellectual development", "implications for how things are taught"],
        "Pedagogy and Styles": ["active learning", "unfamiliar pedagogies", "learning styles", "cooperative learning", "functional regions of the brain"],
    }

    themed_sentences = {k: [] for k in THEMATIC_KEYWORDS.keys()}
    
    # 2. Map keywords to themes for efficient searching
    all_keywords = {word: theme for theme, words in THEMATIC_KEYWORDS.items() for word in words}

    # 3. Categorize each extracted sentence
    for sentence in sentences:
        s_lower = sentence.lower()
        found_theme = None
        
        # Search for a matching keyword
        for keyword, theme in all_keywords.items():
            if keyword in s_lower:
                found_theme = theme
                break
        
        if found_theme:
            # Note: Changed from 'sentence' in 'themed_sentences[found_theme]' for clarity
            if sentence not in themed_sentences[found_theme]: 
                themed_sentences[found_theme].append(sentence)
        else:
            # Fallback for highly-ranked sentences that might not hit a specific keyword
            if "learning" in s_lower and "teach" in s_lower:
                if sentence not in themed_sentences["Foundational Shift"]:
                    themed_sentences["Foundational Shift"].append(sentence)

    # 4. Create the final structured input string
    abstractive_input = {}
    for theme, s_list in themed_sentences.items():
        if s_list:
            # Join the sentences within a theme into a single, cohesive paragraph
            abstractive_input[theme] = " ".join(s_list)
            
    return abstractive_input