# text_classification_model.py
# Full pipeline: preprocessing, TF-IDF, Logistic Regression, evaluation, visuals, and predictions export.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns
import os

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "product_reviews_dataset.csv")
OUTPUT_PRED = os.path.join(BASE_DIR, "model_predictions.csv")
CM_IMAGE = os.path.join(BASE_DIR, "confusion_matrix.png")

# Load dataset
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print(df.head())

# Basic preprocessing
df['review'] = df['review'].astype(str)
df['label'] = df['label'].astype(str)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training - Logistic Regression
model = LogisticRegression(max_iter=1000, solver='liblinear', multi_class='ovr', C=1.0, random_state=42)
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation metrics
acc = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
report = classification_report(y_test, y_pred, zero_division=0)

print("Accuracy: {:.4f}".format(acc))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1-score: {:.4f}".format(f1))
print("\\nClassification Report:\\n", report)

# Confusion matrix
labels = sorted(df['label'].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_IMAGE)
print("Saved confusion matrix to", CM_IMAGE)

# Export predictions
pred_df = pd.DataFrame({
    "review": X_test.values,
    "actual": y_test.values,
    "predicted": y_pred
})
pred_df.to_csv(OUTPUT_PRED, index=False)
print("Saved predictions to", OUTPUT_PRED)
