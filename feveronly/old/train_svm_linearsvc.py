import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
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

# Increase dataset sizes 🚀
train_size = 100000  # Increased from 30,000
test_size = 50000   # Increased from 5,000

# Load datasets with increased sample sizes
train_df = load_jsonl_subset(train_file_path, sample_size=train_size)
dev_df = load_jsonl_subset(dev_file_path, sample_size=test_size)

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
    max_features=100000,  # Increase max features
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

    # Train `LinearSVC` with a single C value
    print(f"\nTraining LinearSVC with C={c_value}...")
    svm_model = LinearSVC(C=c_value, max_iter=3000)  # Increased `max_iter` for better convergence
    svm_model.fit(X_train, y_train)
    print(f"LinearSVC training complete with C={c_value}.")

    # Evaluate `LinearSVC` Model
    print(f"Evaluating LinearSVC with C={c_value}...")
    y_pred = svm_model.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_dev, y_pred, average=None)  # Get per-class scores

    print(f"\nLinearSVC (C={c_value}) - Accuracy: {acc:.4f}")
    print(f"SUPPORTS (1)    - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1-score: {f1[1]:.4f}")
    print(f"REFUTES (0)     - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1-score: {f1[0]:.4f}")
    print(f"NOT ENOUGH INFO (2) - Precision: {precision[2]:.4f}, Recall: {recall[2]:.4f}, F1-score: {f1[2]:.4f}\n")

    # Save results to CSV file
    results_df = pd.DataFrame([[c_value, acc, *precision, *recall, *f1]], 
                              columns=["C", "Accuracy", "Precision_Refutes", "Precision_Supports", "Precision_NEI", 
                                       "Recall_Refutes", "Recall_Supports", "Recall_NEI",
                                       "F1_Refutes", "F1_Supports", "F1_NEI"])
    results_df.to_csv("svm_results_large.csv", mode='a', index=False, header=not pd.io.common.file_exists("svm_results_large.csv"))

print("LinearSVC testing complete. Results saved to svm_results_large.csv.")
