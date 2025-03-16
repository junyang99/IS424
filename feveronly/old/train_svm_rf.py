#SVM (C=0.01) - Accuracy: 0.6150, Precision: 0.5863, Recall: 0.7942, F1-score: 0.6746
#SVM (C=0.1) - Accuracy: 0.6126, Precision: 0.5825, Recall: 0.8078, F1-score: 0.6769
#SVM (C=1.0) - Accuracy: 0.6156, Precision: 0.5871, Recall: 0.7918, F1-score: 0.6742
#SVM (C=10.0) - Accuracy: 0.6129, Precision: 0.5844, Recall: 0.7948, F1-score: 0.6735
#Random Forest - Accuracy: 0.6004, Precision: 0.5646, Recall: 0.8938, F1-score: 0.6921


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

# Load datasets with 20,000 rows for training and 5,000 for testing
train_df = load_jsonl_subset(train_file_path, sample_size=20000)
dev_df = load_jsonl_subset(dev_file_path, sample_size=5000)

# Convert labels: SUPPORTS → 1 (Real), REFUTES → 0 (Fake)
label_map = {"SUPPORTS": 1, "REFUTES": 0}

# Filter out "NOT ENOUGH INFO" labels
train_df = train_df[train_df['label'].isin(label_map)].copy()
dev_df = dev_df[dev_df['label'].isin(label_map)].copy()

# Map labels to binary values
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

print("TF-IDF vectorization complete. Balancing dataset...")

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

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
    precision, recall, f1, _ = precision_recall_fscore_support(y_dev, y_pred, average='binary')
    
    print(f"SVM (C={c_value}) - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}\n")

    # Save results to CSV file
    results_df = pd.DataFrame([[c_value, acc, precision, recall, f1]], 
                              columns=["C", "Accuracy", "Precision", "Recall", "F1-score"])
    results_df.to_csv("svm_results.csv", mode='a', index=False, header=not pd.io.common.file_exists("svm_results.csv"))

print("SVM testing complete. Results saved to svm_results.csv.")

# Train optimized Random Forest
print("\nTraining optimized Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=300,  # More trees
    max_depth=50,  # Limit depth
    min_samples_split=5,  # Prevent overfitting
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("Random Forest training complete.")

# Evaluate Random Forest Model
print("Evaluating Random Forest...")
y_pred_rf = rf_model.predict(X_dev)
acc_rf = accuracy_score(y_dev, y_pred_rf)
precision_rf, recall_rf, f1_rf, _ = precision_recall_fscore_support(y_dev, y_pred_rf, average='binary')

print(f"Random Forest - Accuracy: {acc_rf:.4f}, Precision: {precision_rf:.4f}, Recall: {recall_rf:.4f}, F1-score: {f1_rf:.4f}\n")

# Save Random Forest Results
rf_results_df = pd.DataFrame([[acc_rf, precision_rf, recall_rf, f1_rf]], 
                             columns=["Accuracy", "Precision", "Recall", "F1-score"])
rf_results_df.to_csv("rf_results.csv", mode='w', index=False, header=True)

print("Random Forest results saved to rf_results.csv.")
