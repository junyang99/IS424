#SVM (C=1.0) - Accuracy: 0.4427
#SUPPORTS (1)    - Precision: 0.4166, Recall: 0.5805, F1-score: 0.4851
# REFUTES (0)     - Precision: 0.4971, Recall: 0.4165, F1-score: 0.4533
# NOT ENOUGH INFO (2) - Precision: 0.4307, Recall: 0.3320, F1-score: 0.3750


import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE

# Load JSONL file with a subset for processing
def load_jsonl_subset(file_path, sample_size):
    print(f"Loading dataset from {file_path}...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))  # Parse JSON line
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()[:100]}")
    if not data:
        raise ValueError(f"Error: No valid data found in {file_path}")
    df = pd.DataFrame(data)
    df = df.sample(n=min(sample_size, len(df)), random_state=42)  # Random sampling
    print(f"Loaded {len(df)} rows from {file_path}")
    return df

# File paths
train_file_path = "train (1).jsonl"
dev_file_path = "shared_task_dev.jsonl"

# Load datasets with 30,000 rows for training and 5,000 for testing
train_df = load_jsonl_subset(train_file_path, sample_size=30000)
dev_df = load_jsonl_subset(dev_file_path, sample_size=10000)

# Convert labels: SUPPORTS → 1, REFUTES → 0, NOT ENOUGH INFO → 2
label_map = {"SUPPORTS": 1, "REFUTES": 0, "NOT ENOUGH INFO": 2}

# Map labels to numeric values
train_df['label'] = train_df['label'].map(label_map)
dev_df['label'] = dev_df['label'].map(label_map)

print("Preprocessing complete. Preparing text data...")

# Use claim as text input
train_texts, train_labels = train_df['claim'].tolist(), train_df['label'].tolist()
dev_texts, dev_labels = dev_df['claim'].tolist(), dev_df['label'].tolist()

# TF-IDF Vectorization with optimized settings
print("Applying TF-IDF vectorization...")
vectorizer = TfidfVectorizer(
    sublinear_tf=True, 
    ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
    max_features=50000,  # Increase max features
    stop_words="english"  # Remove stopwords
)
X_train = vectorizer.fit_transform(train_texts)
X_dev = vectorizer.transform(dev_texts)
y_train, y_dev = np.array(train_labels), np.array(dev_labels)

print("TF-IDF vectorization complete.")

# Apply SMOTE to balance the dataset (optional)
use_smote = True  # Set to False if you don't want SMOTE

if use_smote:
    print("Applying SMOTE to balance dataset...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"SMOTE applied. New class distribution:\n{pd.Series(y_train).value_counts()}")

# Ask the user for a C value to test
while True:
    try:
        c_value = float(input("\nEnter a value for C (or type 'exit' to stop testing): "))
    except ValueError:
        print("Exiting SVM testing...")
        break

    # Train SVM with a single C value
    print(f"\nTraining SVM with C={c_value}...")
    svm_model = SVC(kernel='linear', probability=True, C=c_value)
    svm_model.fit(X_train, y_train)
    print(f"SVM training complete with C={c_value}.")

    # Evaluate SVM Model
    print(f"Evaluating SVM with C={c_value}...")
    y_pred = svm_model.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_dev, y_pred, average=None)  # Get per-class scores

    print(f"\nSVM (C={c_value}) - Accuracy: {acc:.4f}")
    print(f"SUPPORTS (1)    - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1-score: {f1[1]:.4f}")
    print(f"REFUTES (0)     - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1-score: {f1[0]:.4f}")
    print(f"NOT ENOUGH INFO (2) - Precision: {precision[2]:.4f}, Recall: {recall[2]:.4f}, F1-score: {f1[2]:.4f}\n")

    # Save results to CSV file
    results_df = pd.DataFrame([[c_value, acc, *precision, *recall, *f1]], 
                              columns=["C", "Accuracy", "Precision_Refutes", "Precision_Supports", "Precision_NEI", 
                                       "Recall_Refutes", "Recall_Supports", "Recall_NEI",
                                       "F1_Refutes", "F1_Supports", "F1_NEI"])
    results_df.to_csv("svm_results.csv", mode='a', index=False, header=not pd.io.common.file_exists("svm_results.csv"))

print("SVM testing complete. Results saved to svm_results.csv.")
