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

# Function to search Wikipedia for evidence
def search_wikipedia(claim):
    try:
        # Extract key entity from claim
        for entity in KNOWN_ENTITIES:
            if entity.lower() in claim.lower():
                wiki_title = KNOWN_ENTITIES[entity]  # Force correct page
                try:
                    page = wikipedia.page(wiki_title, auto_suggest=False)  # Exact match
                    evidence_text = clean_text(page.content[:2000])  # First 2000 chars
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
    print(f"Processing claim: {claim[:50]}...")  # Show first 50 characters of claim
    wiki_title, wiki_text = search_wikipedia(claim)
    
    if not wiki_title:
        print(f"  → No Wikipedia page found.")
        return "NOT ENOUGH INFO", 0.0  # No evidence found, similarity is 0
    
    similarity_score = compute_tfidf_similarity(claim, wiki_text)
    print(f"  → Wikipedia page: {wiki_title}, Similarity: {similarity_score:.2f}")

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

# TF-IDF Vectorization on Claims
print("\nPerforming TF-IDF vectorization...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
X_train_tfidf = vectorizer.fit_transform(train_df['claim'])
X_dev_tfidf = vectorizer.transform(dev_df['claim'])
print("✔ TF-IDF vectorization complete.")

# Combine TF-IDF and Wikipedia similarity score as features
X_train_combined = np.hstack((X_train_tfidf.toarray(), train_df['similarity_score'].values.reshape(-1, 1)))
X_dev_combined = np.hstack((X_dev_tfidf.toarray(), dev_df['similarity_score'].values.reshape(-1, 1)))

# Train XGBoost Model
print("\nTraining XGBoost model...")
xgb_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, n_jobs=-1)
xgb_model.fit(X_train_combined, train_df['label'])
print("✔ XGBoost training complete.")

# Evaluate XGBoost Model
print("\nEvaluating XGBoost model...")
y_pred = xgb_model.predict(X_dev_combined)

# Print Metrics
accuracy = accuracy_score(dev_df['label'], y_pred)
print(f"\n✔ XGBoost Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(dev_df['label'], y_pred, target_names=["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]))
