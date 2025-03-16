#Evaluating XGBoost model...

#XGBoost - Accuracy: 0.6435
#SUPPORTS (1)    - Precision: 0.6022, Recall: 0.8561, F1-score: 0.7071
#REFUTES (0)     - Precision: 0.7468, Recall: 0.4286, F1-score: 0.5446

#SMOTE applied. New class distribution:
#1    21809
#0    21809

import pandas as pd
import json
import numpy as np
import re
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
import multiprocessing

# Load JSONL file
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
    df = df[df['label'].isin(["SUPPORTS", "REFUTES"])].copy()  # Remove "NOT ENOUGH INFO"
    df = df.sample(n=min(sample_size, len(df)), random_state=42)  # Random sampling
    print(f"Loaded {len(df)} rows from {file_path}")
    return df

# File paths
train_file_path = "train (1).jsonl"
dev_file_path = "shared_task_dev.jsonl"

# Load datasets
train_size = 30000  # Reduced from 50,000 to speed up
test_size = 10000

train_df = load_jsonl_subset(train_file_path, sample_size=train_size)
dev_df = load_jsonl_subset(dev_file_path, sample_size=test_size)

# Convert labels: SUPPORTS → 1, REFUTES → 0
label_map = {"SUPPORTS": 1, "REFUTES": 0}
train_df['label'] = train_df['label'].map(label_map)
dev_df['label'] = dev_df['label'].map(label_map)

print("Preprocessing complete. Preparing text data...")

# Extract text and labels
train_texts, train_labels = train_df['claim'].tolist(), train_df['label'].tolist()
dev_texts, dev_labels = dev_df['claim'].tolist(), dev_df['label'].tolist()

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

train_texts = [preprocess_text(text) for text in train_texts]
dev_texts = [preprocess_text(text) for text in dev_texts]

print("Applying TF-IDF vectorization...")

# TF-IDF Vectorization (REDUCED max_features)
vectorizer = TfidfVectorizer(
    sublinear_tf=True, 
    ngram_range=(1, 2),  # Unigrams & bigrams (no trigrams)
    max_features=20000,  # Reduced from 30,000 for efficiency
    stop_words="english"
)
X_train_tfidf = vectorizer.fit_transform(train_texts)
X_dev_tfidf = vectorizer.transform(dev_texts)

print("TF-IDF vectorization complete.")

# Word2Vec Embeddings (REDUCED vector_size)
print("Training Word2Vec model...")
tokenized_train_texts = [text.split() for text in train_texts]
tokenized_dev_texts = [text.split() for text in dev_texts]

w2v_model = Word2Vec(
    sentences=tokenized_train_texts,
    vector_size=50,  # Reduced from 100 to save memory
    window=5,
    min_count=2,
    workers=multiprocessing.cpu_count()
)
print("Word2Vec model training complete.")

# Convert text into Word2Vec vectors (sentence embeddings)
def get_w2v_features(tokenized_texts, model, vector_size=50):
    features = []
    for tokens in tokenized_texts:
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        if len(vectors) > 0:
            features.append(np.mean(vectors, axis=0))  # Take the mean of all word vectors
        else:
            features.append(np.zeros(vector_size))  # Empty text case
    return np.array(features)

X_train_w2v = get_w2v_features(tokenized_train_texts, w2v_model)
X_dev_w2v = get_w2v_features(tokenized_dev_texts, w2v_model)

print("Word2Vec feature extraction complete.")

# Combine TF-IDF and Word2Vec
X_train_combined = np.hstack((X_train_tfidf.toarray(), X_train_w2v))
X_dev_combined = np.hstack((X_dev_tfidf.toarray(), X_dev_w2v))

print(f"Combined feature shape: {X_train_combined.shape}")

# Apply SMOTE (Optional)
use_smote = True
if use_smote:
    print("Applying SMOTE to balance dataset...")
    smote = SMOTE(random_state=42)
    X_train_combined, train_labels = smote.fit_resample(X_train_combined, train_labels)
    print(f"SMOTE applied. New class distribution:\n{pd.Series(train_labels).value_counts()}")

# Train Optimized XGBoost Model
print("\nTraining Optimized XGBoost model...")
xgb_model = XGBClassifier(
    n_estimators=300,  # Reduced from 500 for efficiency
    max_depth=8,  # Lower depth to speed up training
    learning_rate=0.05,  # Adjusted for stable learning
    colsample_bytree=0.7,  # Feature sampling for regularization
    subsample=0.8,  # Row sampling to prevent overfitting
    n_jobs=4  # Limited to 4 CPU cores to prevent overheating
)

xgb_model.fit(X_train_combined, train_labels)
print("XGBoost training complete.")

# Evaluate XGBoost Model
print("\nEvaluating XGBoost model...")
y_pred = xgb_model.predict(X_dev_combined)

acc = accuracy_score(dev_labels, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(dev_labels, y_pred, average=None)

print(f"\nXGBoost - Accuracy: {acc:.4f}")
print(f"SUPPORTS (1)    - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1-score: {f1[1]:.4f}")
print(f"REFUTES (0)     - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1-score: {f1[0]:.4f}\n")

# Save results to CSV file
results_df = pd.DataFrame([[acc, *precision, *recall, *f1]], 
                          columns=["Accuracy", "Precision_Refutes", "Precision_Supports",
                                   "Recall_Refutes", "Recall_Supports",
                                   "F1_Refutes", "F1_Supports"])
results_df.to_csv("xgboost_optimized_results.csv", mode='w', index=False, header=True)

print("XGBoost results saved to xgboost_optimized_results.csv.")
