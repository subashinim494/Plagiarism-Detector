import nltk
nltk.download("stopwords") 
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import string
from nltk.corpus import stopwords
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle



data = pd.read_csv("dataset.csv")  
print(data.head())



def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

data["source_text"] = data["source_text"].astype(str).apply(preprocess_text)
data["plagiarized_text"] = data["plagiarized_text"].astype(str).apply(preprocess_text)


combined_text = data["source_text"] + " " + data["plagiarized_text"]
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(combined_text)

y = data["label"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = SVC(kernel="linear", random_state=42, probability=True) 
model.fit(X_train, y_train)



y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", cm)



pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf_vectorizer, open("tfidf_vectorizer.pkl", "wb"))


model = pickle.load(open("model.pkl", "rb"))
tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))


def detect(input_text1, input_text2):
    """Check if suspect text is plagiarized from source text"""
    input_text1 = preprocess_text(input_text1)
    input_text2 = preprocess_text(input_text2)
    combined = input_text1 + " " + input_text2
    vectorized = tfidf_vectorizer.transform([combined])
    result = model.predict(vectorized)
    return "Plagiarism Detected" if result[0] == 1 else "No Plagiarism"



source = "Researchers have discovered a new species of butterfly in the Amazon rainforest."
suspect = "A new species of butterfly was discovered in the Amazon rainforest by researchers."

print(detect(source, suspect))  
