# IS424
This project is a Fake News Detector that utilizes Natural Language Processing (NLP) and Machine Learning to classify news articles as either Real or Fake. The model is trained on labeled datasets of fake and true news articles and is deployed via a Streamlit web application. Users can enter a news title and content.

# Features

Allows users to enter a news title and content.

Uses TF-IDF Vectorization to transform text data into numerical features.

Trained using Multinomial Naïve Bayes (MultinomialNB).

Outputs a prediction: "True News" or "Fake News".

Interactive UI powered by Streamlit.

# Dataset

The dataset consists of two CSV files:

Fake.csv - Contains fake news articles.

True.csv - Contains real news articles.

Each dataset has the following columns:

title - The headline of the news article.

text - The body of the news article.

subject - The category of the news (e.g., politics, world news).

date - The publication date.

label - A binary value (0 = Fake, 1 = True).

The dataset contains two types of articles: Fake News and Real News. This dataset was collected from real-world sources; the truthful articles were obtained by crawling articles from Reuters.com (a well-known news website). The fake news articles were collected from unreliable websites that were flagged by Politifact (a fact-checking organization in the USA) and Wikipedia. The dataset contains different types of articles on different topics; however, the majority of articles focus on political and world news topics.

# Installation

1. Clone the Repository

git clone
cd fake-news-detector

2. Install Dependencies

Make sure you have Python installed, then install the required packages:

pip install -r requirements.txt

3. Train the Model (if not already trained)

Run the Jupyter Notebook to train and save the model:

jupyter notebook eda.ipynb

This will generate model.pkl and vectorizer.pkl.

4. Run the Streamlit App

streamlit run app.py

# How It Works

User Input: Enter a title and news content.

Text Processing: The input is vectorized using TF-IDF.

Prediction: The trained Naïve Bayes model classifies the news as True or Fake.

Output: The result is displayed on the UI.

Model Performance

Accuracy: ~93%

Evaluation: Model performance is visualized using a confusion matrix and classification report.

# Testing Fake News Link

Example of a fake news fact-check: Denzel Washington Supports Trump? (False)

# Future Improvements

Implement deep learning models (LSTMs, Transformers).

Allow users to input URLs for automated fact-checking.

Provide explanations for predictions.

# License

This project is open-source and available under the MIT License.
