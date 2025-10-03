# Fake News Detector for Students
# Author: Naveen S S

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def train_model():
    # Load dataset (make sure 'news.csv' is in the repo)
    data = pd.read_csv("news.csv")

    # Split into features and labels
    X = data['text']
    y = data['label']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Text vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train model
    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model trained with Accuracy: {round(acc*100,2)}%\n")

    return model, vectorizer

def predict_news(model, vectorizer, text):
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    return prediction[0]

if __name__ == "__main__":
    model, vectorizer = train_model()
    while True:
        news_input = input("Enter a news article/headline (or type 'exit'): ")
        if news_input.lower() == "exit":
            break
        result = predict_news(model, vectorizer, news_input)
        print("Result:", "REAL News ✅" if result == "REAL" else "FAKE News ❌")
