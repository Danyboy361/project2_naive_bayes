# Project 2: Naive Bayes - Spam Detection (Simple Version)

import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --------------------------------------------------
# 1. Get file location
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "spam.csv")

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------
df = pd.read_csv(csv_path, encoding="latin-1")

# Keep only useful columns
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# --------------------------------------------------
# 3. Split data
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# --------------------------------------------------
# 4. Text vectorization
# --------------------------------------------------
vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --------------------------------------------------
# 5. Train Naive Bayes model
# --------------------------------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# --------------------------------------------------
# 6. Predictions
# --------------------------------------------------
y_pred = model.predict(X_test_vec)

# --------------------------------------------------
# 7. Evaluation
# --------------------------------------------------
print("\nModel Performance:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label="spam"))
print("Recall   :", recall_score(y_test, y_pred, pos_label="spam"))
print("F1-score :", f1_score(y_test, y_pred, pos_label="spam"))

# --------------------------------------------------
# 8. Save model and vectorizer
# --------------------------------------------------
joblib.dump(model, os.path.join(BASE_DIR, "spam_model.pkl"))
joblib.dump(vectorizer, os.path.join(BASE_DIR, "vectorizer.pkl"))

print("\nModel and vectorizer saved successfully.")
