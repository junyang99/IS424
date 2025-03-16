import pandas as pd
import re
import string
import time
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Track execution time
start_time = time.time()

# Load datasets
true_df = pd.read_csv('True.csv')  # Change path if needed
fake_df = pd.read_csv('Fake.csv')  # Change path if needed

# Assign labels
true_df['label'] = 1  # Real News
fake_df['label'] = 0  # Fake News

# Combine datasets
df = pd.concat([true_df, fake_df], axis=0).reset_index(drop=True)

# Identify text column
text_column = 'text' if 'text' in df.columns else df.columns[0]

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)  # Remove numbers
    return text

df[text_column] = df[text_column].apply(clean_text)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[text_column], df['label'], test_size=0.2, random_state=42)

# Convert text to numerical format using TF-IDF
# Convert text to numerical format using TF-IDF (removing 'reuters')
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, token_pattern=r'\b(?!reuters\b)\w+\b')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Define the XGBoost classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Define hyperparameter grid (reduced search space for faster execution)
param_grid = {
    'n_estimators': [100, 200, 300],  # Increased estimators
    'max_depth': [3, 5, 7],  # More depth options
    'learning_rate': [0.01, 0.05, 0.1],  # More learning rates
    'subsample': [0.8, 1.0],  
    'colsample_bytree': [0.8, 1.0],
}

# Perform RandomizedSearchCV with limited iterations (5 instead of 10+ for speed)
random_search = RandomizedSearchCV(
    xgb_model, param_distributions=param_grid, 
    n_iter=10, scoring='accuracy', cv=3, 
    verbose=2, random_state=42, n_jobs=-1
)

random_search.fit(X_train_tfidf, y_train)

# Best model from RandomizedSearchCV
best_xgb_model = random_search.best_estimator_

# Make predictions
y_pred = best_xgb_model.predict(X_test_tfidf)

# Evaluate the model
report = classification_report(y_test, y_pred)

print("\nBest Parameters:", random_search.best_params_)
print("\nClassification Report:\n", report)

# Print execution time
end_time = time.time()
print("\nExecution Time: {:.2f} minutes".format((end_time - start_time) / 60))

# 🔹 NEW SECTION: Find the Top 10 Most Important Words
print("\n🔹 Extracting Top 10 Words the Model is Overfitting To...\n")

# Get feature importance from the trained XGBoost model
feature_importances = best_xgb_model.feature_importances_

# Get feature names from the TF-IDF vectorizer
feature_names = vectorizer.get_feature_names_out()

# Sort feature importances in descending order
sorted_indices = np.argsort(feature_importances)[::-1]
top_n = 10  # Number of top words to display

# Extract the top N important words
top_features = [(feature_names[i], feature_importances[i]) for i in sorted_indices[:top_n]]

# Convert to DataFrame for better visualization
top_features_df = pd.DataFrame(top_features, columns=["Word", "Importance"])

# Print results
print("\n🔹 Top 10 Words the Model is Overfitting To:\n")
print(top_features_df)
