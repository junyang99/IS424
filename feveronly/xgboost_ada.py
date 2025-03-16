#📊 Classification Report (F1-Score Table):
#              precision    recall  f1-score      support
#REFUTES        0.789129  0.470447  0.589474  3333.000000
#SUPPORTS       0.622783  0.874287  0.727409  3333.000000
#accuracy       0.672367  0.672367  0.672367     0.672367
#macro avg      0.705956  0.672367  0.658441  6666.000000
#weighted avg   0.705956  0.672367  0.658441  6666.000000

import json
import numpy as np
import pandas as pd
import re
import fasttext
import fasttext.util
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import ADASYN  # Using ADASYN instead of SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Load FEVER dataset
def load_fever_data(file_path, sample_size=50000):  
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
    df = df[df['label'].isin(["SUPPORTS", "REFUTES"])].copy()  # Remove "NOT ENOUGH INFO"
    print(f"✔ Loaded {len(df)} rows from {file_path} (Only SUPPORTS & REFUTES)")
    return df

# Load datasets
train_file_path = "train (1).jsonl"
dev_file_path = "shared_task_dev.jsonl"

train_df = load_fever_data(train_file_path, sample_size=50000)  
dev_df = load_fever_data(dev_file_path, sample_size=10000)  

# Convert labels to binary values (SUPPORTS = 1, REFUTES = 0)
label_map = {"SUPPORTS": 1, "REFUTES": 0}
train_df['label'] = train_df['label'].map(label_map)
dev_df['label'] = dev_df['label'].map(label_map)

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

train_df["clean_claim"] = train_df["claim"].apply(preprocess_text)
dev_df["clean_claim"] = dev_df["claim"].apply(preprocess_text)

# TF-IDF Vectorization
print("\nPerforming TF-IDF vectorization...")
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=30000,  
    ngram_range=(1,3),  
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(train_df['clean_claim'])
X_dev_tfidf = vectorizer.transform(dev_df['clean_claim'])
print("✔ TF-IDF vectorization complete.")

# Load FastText Model for Word Embeddings
print("\nLoading FastText embeddings...")
fasttext.util.download_model('en', if_exists='ignore')  
ft_model = fasttext.load_model('cc.en.300.bin')

# Convert text into FastText embeddings
def get_fasttext_features(texts, model):
    features = []
    for text in texts:
        words = text.split()
        vectors = [model.get_word_vector(word) for word in words if word in model.words]
        if len(vectors) > 0:
            features.append(np.mean(vectors, axis=0))  
        else:
            features.append(np.zeros(300))  
    return np.array(features)

X_train_ft = get_fasttext_features(train_df['clean_claim'], ft_model)
X_dev_ft = get_fasttext_features(dev_df['clean_claim'], ft_model)

# Combine TF-IDF + FastText
X_train_combined = np.hstack((X_train_tfidf.toarray(), X_train_ft))
X_dev_combined = np.hstack((X_dev_tfidf.toarray(), X_dev_ft))

# Apply ADASYN for Class Balancing
print("\nApplying ADASYN for class balancing (Binary Classes)...")
adasyn = ADASYN(random_state=42)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_combined, train_df['label'])
print(f"✔ ADASYN applied successfully. New class distribution:\n{pd.Series(y_train_resampled).value_counts()}")

# Train Optimized XGBoost Model (Binary)
print("\nTraining Optimized XGBoost model...")
xgb_model = XGBClassifier(
    n_estimators=300,  
    max_depth=10,  
    learning_rate=0.05,  
    colsample_bytree=0.9,  
    subsample=0.9,  
    n_jobs=-1,  
    scale_pos_weight=1.0,  
    objective='binary:logistic'
)

xgb_model.fit(X_train_resampled, y_train_resampled)
print("✔ XGBoost training complete.")

# Evaluate Model
print("\nEvaluating XGBoost model...")
y_pred = xgb_model.predict(X_dev_combined)

# Generate classification report
report_dict = classification_report(dev_df['label'], y_pred, target_names=["REFUTES", "SUPPORTS"], output_dict=True)

# Convert to DataFrame and Save
report_df = pd.DataFrame(report_dict).transpose()
print("\n📊 Classification Report (F1-Score Table):")
print(report_df)
report_df.to_excel("XGBoost_Binary_Classification_Report_ADASYN.xlsx")

# Save claim predictions
dev_df["predicted_label"] = y_pred
dev_df.to_excel("XGBoost_Binary_Predictions_ADASYN.xlsx", index=False)
print("✔ Predictions saved successfully!")
