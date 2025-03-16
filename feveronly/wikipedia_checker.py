import wikipedia
import re
from wikipedia.exceptions import PageError, DisambiguationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to clean Wikipedia text
def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)  # Remove citations [1], [2]...
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Predefined entity mapping for important figures
KNOWN_ENTITIES = {
    "Barack Obama": "Barack Obama",
    "Elon Musk": "Elon Musk",
    "Eiffel Tower": "Eiffel Tower",
    "NASA": "NASA",
}

# Function to search Wikipedia for evidence
def search_wikipedia(claim):
    try:
        # Extract key entity from claim
        for entity in KNOWN_ENTITIES:
            if entity.lower() in claim.lower():
                wiki_title = KNOWN_ENTITIES[entity]  # Force correct page
                try:
                    page = wikipedia.page(wiki_title, auto_suggest=False)  # Exact match
                    evidence_text = clean_text(page.content[:2000])  # Use first 2000 chars
                    return wiki_title, evidence_text
                except PageError:
                    return None, f"Page not found for {entity}."

        # Otherwise, do a normal Wikipedia search
        search_results = wikipedia.search(claim, results=3)
        if not search_results:
            return None, "No relevant Wikipedia page found."

        # Try best match
        for result in search_results:
            try:
                page = wikipedia.page(result, auto_suggest=True)
                evidence_text = clean_text(page.content[:2000])
                return result, evidence_text
            except DisambiguationError:
                continue  # Skip ambiguous results
            except PageError:
                continue  # Skip invalid pages
        
        return None, "No suitable Wikipedia page found."

    except Exception as e:
        return None, f"Error: {e}"

# Function to compute TF-IDF similarity
def compute_tfidf_similarity(claim, evidence_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([claim, evidence_text])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity_score

# Function to check claim with Wikipedia evidence using TF-IDF
def fact_check_claim(claim):
    wiki_title, wiki_text = search_wikipedia(claim)
    
    if not wiki_title:
        return "NOT ENOUGH INFO", wiki_text  # No evidence found
    
    # Compute similarity between claim and Wikipedia evidence
    similarity_score = compute_tfidf_similarity(claim, wiki_text)

    # Classification based on similarity score
    if similarity_score >= 0.3:  # High similarity → Supports claim
        return "SUPPORTS", f"Found in Wikipedia: {wiki_title} (Similarity: {similarity_score:.2f})"
    else:  # Low similarity → Likely refuted
        return "REFUTES", f"Contradicted by Wikipedia: {wiki_title} (Similarity: {similarity_score:.2f})"

# Test Example Claims
claims = [
    "Elon Musk is the CEO of Apple Inc.",
    "The Eiffel Tower was built in 1889.",
    "Barack Obama was born in Kenya.",
    "NASA confirmed an asteroid is going to hit Earth in 2025."
]

# Run fact-checking
for claim in claims:
    label, explanation = fact_check_claim(claim)
    print(f"\n🔍 Claim: {claim}")
    print(f"🟢 Verdict: {label}")
    print(f"📖 Explanation: {explanation}")
