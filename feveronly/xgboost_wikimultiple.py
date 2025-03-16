import wikipedia
import re
import json
import numpy as np
import pandas as pd
from wikipedia.exceptions import PageError, DisambiguationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Function to clean Wikipedia text
def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)  # Remove citations
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Predefined entity mapping for important figures
KNOWN_ENTITIES = {
    "Barack Obama": "Barack Obama",
    "Elon Musk": "Elon Musk",
    "Eiffel Tower": "Eiffel Tower",
    "NASA": "NASA",
}

# Function to search multiple Wikipedia pages
def search_wikipedia(claim):
    try:
        # Extract key entity from claim
        for entity in KNOWN_ENTITIES:
            if entity.lower() in claim.lower():
                wiki_title = KNOWN_ENTITIES[entity]  # Force correct page
                try:
                    page = wikipedia.page(wiki_title, auto_suggest=False)  # Exact match
                    evidence_text = clean_text(page.content[:5000])  # Extract first 5000 characters
                    return [(wiki_title, evidence_text)]
                except PageError:
                    return []

        # Otherwise, do a normal Wikipedia search (Top 3 pages)
        search_results = wikipedia.search(claim, results=3)
        if not search_results:
            return []

        # Fetch content from multiple pages
        results = []
        for result in search_results:
            try:
                page = wikipedia.page(result, auto_suggest=True)
                evidence_text = clean_text(page.content[:5000])
                results.append((result, evidence_text))
            except DisambiguationError:
                continue  # Skip ambiguous results
            except PageError:
                continue  # Skip invalid pages

        return results

    except Exception as e:
        return []

# Function to compute TF-IDF similarity for multiple pages
def compute_best_similarity(claim, evidence_texts):
    if not evidence_texts:
        return 0.0  # No evidence found

    vectorizer = TfidfVectorizer(stop_words="english")
    
    # Compute similarity for each Wikipedia page and return the highest one
    max_similarity = 0.0
    for evidence_text in evidence_texts:
        tfidf_matrix = vectorizer.fit_transform([claim, evidence_text])
        similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        max_similarity = max(max_similarity, similarity_score)
    
    return max_similarity

# Function to check claim with Wikipedia evidence using TF-IDF
def fact_check_claim(claim):
    print(f"Processing claim: {claim[:50]}...")  # Show first 50 characters of claim
    wiki_results = search_wikipedia(claim)
    
    if not wiki_results:
        print(f"  → No Wikipedia page found.")
        return "NOT ENOUGH INFO", 0.0  # No evidence found, similarity is 0
    
    # Compute the highest similarity score
    similarity_score = compute_best_similarity(claim, [text for _, text in wiki_results])

    print(f"  → Checked {len(wiki_results)} Wikipedia pages, Best Similarity: {similarity_score:.2f}")

    if similarity_score >= 0.3:
        return "SUPPORTS", similarity_score
    else:
        return "REFUTES", similarity_score

# Function to load FEVER dataset
def load_fever_data(file_path, sample_size=5000):
    print(f"Loading dataset from {file_path}...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i >= sample_size:
                break
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line at index {i}")
    
    df = pd.DataFrame(data)
    df = df[df['label'].isin(["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"])].copy()
    print(f"✔ Loaded {len(df)} rows from {file_path}")
    return df

# Load training and development datasets
train_file_path = "train (1).jsonl"
dev_file_path = "shared_task_dev.jsonl"

train_df = load_fever_data(train_file_path, sample_size=5000)
dev_df = load_fever_data(dev_file_path, sample_size=2000)

# Convert labels to numerical values
label_map = {"SUPPORTS": 1, "REFUTES": 0, "NOT ENOUGH INFO": 2}
train_df['label'] = train_df['label'].map(label_map)
dev_df['label'] = dev_df['label'].map(label_map)

# Apply Wikipedia fact-checking to claims
print("\nApplying Wikipedia fact-checking... (This will take time)")
train_df['fact_check_result'], train_df['similarity_score'] = zip(*train_df['claim'].map(fact_check_claim))
dev_df['fact_check_result'], dev_df['similarity_score'] = zip(*dev_df['claim'].map(fact_check_claim))

# Train XGBoost Model
print("\nTraining XGBoost model...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
X_train_tfidf = vectorizer.fit_transform(train_df['claim'])
X_dev_tfidf = vectorizer.transform(dev_df['claim'])

X_train_combined = np.hstack((X_train_tfidf.toarray(), train_df['similarity_score'].values.reshape(-1, 1)))
X_dev_combined = np.hstack((X_dev_tfidf.toarray(), dev_df['similarity_score'].values.reshape(-1, 1)))

xgb_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, n_jobs=-1)
xgb_model.fit(X_train_combined, train_df['label'])
print("✔ XGBoost training complete.")

# Evaluate Model
print("\nEvaluating XGBoost model...")
y_pred = xgb_model.predict(X_dev_combined)

accuracy = accuracy_score(dev_df['label'], y_pred)
print(f"\n✔ XGBoost Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(dev_df['label'], y_pred))
