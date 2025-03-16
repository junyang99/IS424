#changing sample size or max_feature does not improve model accuracy only does so by 0.01 percent
# utilised smote to balance classes, preventing the model fromfavouring one label. but results do not change
#Model Accuracy: 0.4414 (final score after smote)

#Classification Report:
#              precision    recall  f1-score   support

#           0       0.50      0.42      0.45      1670
 #          1       0.42      0.56      0.48      1686
  #         2       0.41      0.34      0.37      1644

#    accuracy                           0.44      5000
#   macro avg       0.44      0.44      0.44      5000
#weighted avg       0.44      0.44      0.44      5000

import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load dataset with error handling
def load_jsonl_subset(file_path, sample_size=20000):
    print(f"Loading dataset from {file_path}...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            if not line.startswith("{"):  # Skip lines that don't start with JSON object
                print(f"Skipping invalid JSON at line {i + 1}: {line[:100]}")
                continue
            try:
                json_data = json.loads(line)
                data.append(json_data)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at line {i + 1}: {line[:100]}")
                continue
    if not data:
        raise ValueError(f"No valid data found in {file_path}")
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} rows from {file_path}")
    return df  # Return DataFrame instead of list

# Convert dataset into a Pandas DataFrame
def preprocess_data(df):
    print("Preprocessing data...")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Expected input data as a DataFrame")

    # Ensure required columns exist
    if "claim" not in df.columns or "label" not in df.columns:
        raise ValueError("Missing 'claim' or 'label' columns in the dataset")

    # Keep all three labels
    label_mapping = {"SUPPORTS": 1, "REFUTES": 0, "NOT ENOUGH INFO": 2}

    # Apply mapping
    df["label"] = df["label"].map(label_mapping)

    print("Data preprocessing complete. Labels retained (3 classes).")
    return df[["claim", "label"]]  # Return only required columns

# File paths
train_file_path = "train (1).jsonl"
test_file_path = "shared_task_dev.jsonl"

# Load train and test data with error handling and sampling
print("\nStarting data loading process...")
train_df = load_jsonl_subset(train_file_path).sample(n=30000, random_state=42)
test_df = load_jsonl_subset(test_file_path).sample(n=5000, random_state=42)

# Preprocess data
train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

# Convert text claims into numerical features using TF-IDF
print("\nApplying TF-IDF vectorization...")
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X_train = vectorizer.fit_transform(train_df["claim"])
X_test = vectorizer.transform(test_df["claim"])
y_train = train_df["label"]
y_test = test_df["label"]
print("TF-IDF vectorization complete.")

# Apply SMOTE to balance the dataset
print("\nApplying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print(f"SMOTE applied. New class distribution:\n{pd.Series(y_train).value_counts()}")

# Train Logistic Regression model
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=200, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training complete.")

# Make predictions
print("\nMaking predictions on the test set...")
y_pred = model.predict(X_test)
print("Predictions complete.")

# Evaluate the model
print("\nEvaluating the model...")
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nScript execution complete. ✅")
