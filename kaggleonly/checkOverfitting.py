import numpy as np
import pandas as pd

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
