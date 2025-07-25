import pandas as pd
import numpy as np
import string
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42

print("Loading data...")
# Load the dataset
df = pd.read_csv('data.csv', quoting=3, encoding='utf-8', on_bad_lines='warn')

print(f"Dataset shape: {df.shape}")

# Data preprocessing and cleaning
print("Preprocessing and cleaning data...")
# Remove rows with missing values in statement or status
df = df.dropna(subset=['statement', 'status'])

# Remove rows where status is empty string or whitespace
df = df[df['status'].astype(str).str.strip() != '']

# Filter to keep only the main categories to avoid class imbalance issues
main_categories = ['Anxiety', 'Normal', 'Depression', 'Suicidal', 'Bipolar']
df = df[df['status'].isin(main_categories)]

print(f"Dataset shape after cleaning: {df.shape}")
print(f"Status distribution:\n{df['status'].value_counts()}")

# Text preprocessing
df['statement'] = df['statement'].astype(str).str.lower()
df['statement'] = df['statement'].str.replace('[{}]'.format(string.punctuation), '', regex=True)

print("Creating TF-IDF features...")
# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf_vectorizer.fit_transform(df['statement'])
y = df['status']

print("Splitting data...")
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

print("Training model...")
# Train the Naive Bayes model (no need for additional cleaning since we already cleaned the data)
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
print("Evaluating model...")
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nModel Performance:")
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')

print("\nSaving model and vectorizer...")
# Save both the model and the TF-IDF vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("Model and TF-IDF vectorizer saved successfully!")

# Test the prediction function
def predict_status(statement):
    """
    Predicts the mental health status of a given statement.
    
    Args:
        statement: The input statement as a string.
    
    Returns:
        The predicted mental health status.
    """
    # Preprocess the statement (lowercase, remove punctuation)
    statement = statement.lower()
    statement = re.sub('[{}]'.format(string.punctuation), '', statement)
    
    # Vectorize the statement
    statement_vectorized = tfidf_vectorizer.transform([statement])
    
    # Predict the status
    predicted_status = model.predict(statement_vectorized)
    
    return predicted_status[0]

# Test with some examples
print("\nTesting predictions:")
test_statements = [
    "I feel like something bad is going to happen all the time",
    "I had a pretty good day today",
    "I can't stop worrying about things going wrong",
    "I feel happy and content"
]

for stmt in test_statements:
    prediction = predict_status(stmt)
    print(f"Statement: '{stmt}' -> Predicted: {prediction}")
