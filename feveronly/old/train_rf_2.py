import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD

# Load dataset
def load_jsonl_subset(file_path, sample_size):
    print(f"Loading dataset from {file_path}...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()[:100]}")
    if not data:
        raise ValueError(f"Error: No valid data found in {file_path}")
    df = pd.DataFrame(data)
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    print(f"Loaded {len(df)} rows from {file_path}")
    return df

# File paths
train_file_path = "train (1).jsonl"
dev_file_path = "shared_task_dev.jsonl"

# Load datasets
train_df = load_jsonl_subset(train_file_path, sample_size=50000)
dev_df = load_jsonl_subset(dev_file_path, sample_size=10000)

# Convert labels (Binary: SUPPORTS → 1, REFUTES → 0)
label_map = {"SUPPORTS": 1, "REFUTES": 0}

# Filter out "NOT ENOUGH INFO" and apply mapping
train_df = train_df[train_df['label'].isin(label_map)].copy()
dev_df = dev_df[dev_df['label'].isin(label_map)].copy()
train_df['label'] = train_df['label'].map(label_map)
dev_df['label'] = dev_df['label'].map(label_map)

print("Preprocessing complete. Preparing text data...")

# Use claims as text input
train_texts, train_labels = train_df['claim'].tolist(), train_df['label'].tolist()
dev_texts, dev_labels = dev_df['claim'].tolist(), dev_df['label'].tolist()

# TF-IDF Vectorization
print("Applying TF-IDF vectorization...")
tfidf_vectorizer = TfidfVectorizer(
    sublinear_tf=True, 
    ngram_range=(1, 3), 
    max_features=50000, 
    stop_words="english"
)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
X_dev_tfidf = tfidf_vectorizer.transform(dev_texts)
print("TF-IDF vectorization complete.")

# Word2Vec Embeddings
print("Training Word2Vec model...")
tokenized_texts = [text.split() for text in train_texts]
word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=2, workers=4)

def get_avg_word2vec(text, model, vector_size):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) == 0:  # Handle empty cases
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)

X_train_w2v = np.array([get_avg_word2vec(text, word2vec_model, 100) for text in train_texts])
X_dev_w2v = np.array([get_avg_word2vec(text, word2vec_model, 100) for text in dev_texts])
print("Word2Vec embeddings complete.")

# Combine TF-IDF + Word2Vec Features
X_train_combined = np.hstack((X_train_tfidf.toarray(), X_train_w2v))
X_dev_combined = np.hstack((X_dev_tfidf.toarray(), X_dev_w2v))

# Apply SVD first, then SMOTE
print("Applying SVD for dimensionality reduction...")
svd = TruncatedSVD(n_components=300)
X_train_combined = svd.fit_transform(X_train_combined)
X_dev_combined = svd.transform(X_dev_combined)
print("Dimensionality reduction complete.")

# Apply SMOTE
print("Applying SMOTE to balance dataset...")
y_train = np.array(train_labels)
smote = SMOTE(random_state=42)
X_train_combined, y_train = smote.fit_resample(X_train_combined, y_train)
print(f"SMOTE applied. New class distribution:\n{pd.Series(y_train).value_counts()}")

# Train Random Forest
print("\nTraining optimized Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=500,  
    max_depth=70,  
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_combined, y_train)
print("Random Forest training complete.")

# Evaluate Random Forest Model
print("Evaluating Random Forest...")
y_dev = np.array(dev_labels)
y_pred_rf = rf_model.predict(X_dev_combined)
acc_rf = accuracy_score(y_dev, y_pred_rf)
precision_rf, recall_rf, f1_rf, _ = precision_recall_fscore_support(y_dev, y_pred_rf, average=None)

print(f"\nRandom Forest - Accuracy: {acc_rf:.4f}")
print(f"SUPPORTS (1)    - Precision: {precision_rf[1]:.4f}, Recall: {recall_rf[1]:.4f}, F1-score: {f1_rf[1]:.4f}")
print(f"REFUTES (0)     - Precision: {precision_rf[0]:.4f}, Recall: {recall_rf[0]:.4f}, F1-score: {f1_rf[0]:.4f}")

print("Random Forest results saved.")
