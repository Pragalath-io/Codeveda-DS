import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

nltk.download('stopwords')

data = {
    'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'ham', 'spam', 'ham'],
    'message': [
        'Hey, are we still on for tonight?',
        'Congratulations! You have won a FREE entry to our $1000 prize draw. Text WIN to 8888.',
        'Can you pick up some groceries on your way home?',
        'URGENT! Your account has been suspended. Please click this link to verify your identity.',
        'Just wanted to say I had a great time.',
        'FREE Viagra! Best prices guaranteed. Click here now!',
        'See you later alligator!',
        'What time is the meeting tomorrow?',
        'You have been selected for a secret shopping survey. Earn $500. Reply YES.',
        'Sounds good, I will be there.'
    ]
}
df = pd.DataFrame(data)

print("--- Sample Dataset ---")
print(df)
print("\n")

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.lower().split()
    stemmed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(stemmed_tokens)

df['processed_message'] = df['message'].apply(preprocess_text)
print("--- Preprocessed Dataset ---")
print(df[['label', 'processed_message']])
print("\n")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_message'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
print("Naive Bayes model trained successfully.\n")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\n")

print("--- Testing with New Messages ---")
new_messages = [
    "Can you call me back?",
    "Free prize money, click this link"
]
processed_new = [preprocess_text(msg) for msg in new_messages]
new_vectors = vectorizer.transform(processed_new)
predictions = model.predict(new_vectors)

for msg, pred in zip(new_messages, predictions):
    print(f"Message: '{msg}' -> Prediction: {pred.upper()}")