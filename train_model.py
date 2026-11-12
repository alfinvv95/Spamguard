
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load our dataset
df = pd.read_csv("dataset/emails.csv")

# Remove empty email records
df = df.dropna(subset=["text", "label"])

X = df["text"].astype(str)
y = df["label"].astype(str)

# Divide data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert email text into numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create and train the AI model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Test the model
predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save the trained model and vectorizer
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")
