# Project 2: Naive Bayes – Spam Detection

## Project Overview
This project implements a **Naive Bayes classification model** to detect
whether an SMS message is **Spam** or **Ham (Not Spam)**.
The project demonstrates text preprocessing, probabilistic classification,
model evaluation, and model persistence.

## Dataset
- **Name:** SMS Spam Collection Dataset
- **Source:** Kaggle
- **File:** spam.csv

The dataset contains labeled SMS messages classified as spam or ham.

## Approach Used
- Text preprocessing and vectorization using **Bag of Words**
- Classification using **Multinomial Naive Bayes**

## Tools & Libraries
- Python
- pandas
- scikit-learn
- joblib

## Text Preprocessing
- Removal of unnecessary columns
- Tokenization and vectorization using `CountVectorizer`
- Stop-word removal

## Model Used
- Multinomial Naive Bayes

## Evaluation Metrics
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

### Sample Results
- Accuracy: ~0.98
- Precision: ~0.96
- Recall: ~0.92
- F1-score: ~0.94

## How to Run the Project
1. Place `project2.py` and `spam.csv` in the same folder.
2. Open Command Prompt / PowerShell.
3. Run:
   ```bash
   python project2.py
