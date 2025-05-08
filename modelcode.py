# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle

# Example Data Preparation
# (In your real use case, you would replace this with your actual dataset)
data = {
    'text': [
        "This is a good product",
        "I hated the experience",
        "Fantastic service and friendly staff",
        "Worst service ever",
        "Loved the ambiance and food",
        "Not worth the money",
    ],
    'label': [1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Step 1: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# Step 2: Train SVM Model
model = SVC(probability=True)  # (probability=True if you want predict_proba, optional)
model.fit(X_train_tfidf, y_train)

# Step 3: Save the TF-IDF Vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Step 4: Save the Trained SVM Model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# ------------

# Step 5: (Later) Load the Saved Model and Vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Step 6: Make a Prediction
sample_text = ["The food was amazing"]
sample_text_tfidf = loaded_vectorizer.transform(sample_text)
prediction = loaded_model.predict(sample_text_tfidf)

print("Predicted label:", prediction[0])
