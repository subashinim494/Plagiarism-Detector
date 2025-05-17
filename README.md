# 📄 Plagiarism Detector using 

A machine learning web application that detects plagiarism in text using Natural Language Processing (NLP). Built using Python, Flask, and scikit-learn, this app allows users to input text and receive an evaluation of potential plagiarism against a dataset.

---

## 🚀 Features

- 📚 Compares input text against a dataset to detect similarities  
- 🔤 Uses TF-IDF vectorization for text preprocessing  
- 🤖 Powered by a trained **Support Vector Machine (SVM)** model  
- 🧠 Pre-trained model and vectorizer (`model.pkl`, `tfidf_vectorizer.pkl`)  
- 🌐 Simple and clean web interface using Flask

---

## 🌐 Live Demo

👉 [Click here to view the deployed app](https://plagiarism-detector-kn5a.onrender.com/)

---

## 🖼️ Screenshots
![Home Page](Screenshots/Screenshot%202025-05-17%20195539.png)
 
![Input ](Screenshots/Screenshot%202025-05-17%20195626.png)

![Result](Screenshots/Screenshot%202025-05-17%20195648.png)

---

## 🛠️ Tech Stack

- **Python**  
- **Flask**  
- **scikit-learn (SVM Classifier)**  
- **HTML/CSS (Jinja2 templates)**  
- **Pandas, NumPy**

---

## 🗂️ Project Structure

```bash
Plagiarism-Detector/
│
├── app.py # Flask application
├── modelcode.py # Model training script (SVM)
├── model.pkl # Trained SVM model
├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
├── cleaned_plagiarism_dataset.csv
├── templates/
│ └── index.html # Web UI
├── screenshots/
│ ├── homepage.png # UI screenshot
│ └── prediction.png # Result screenshot
├── requirements.txt # Project dependencies
├── Procfile # For deployment on Heroku
└── README.md # Project documentation
 ```
---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/SuhainaFathimaM/Plagiarism-Detector.git
cd Plagiarism-Detector
```
### 2. (Optional) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the application
```bash
python app.py
Then open your browser and go to: http://127.0.0.1:5000/
```
--- 
## 📊 Dataset
The dataset used for training and prediction is cleaned_plagiarism_dataset.csv, which includes preprocessed textual data with plagiarism labels.

---
## 🧠 Model Training
To retrain the SVM model and generate the required .pkl files:

```bash

python modelcode.py
```
This will output:

- model.pkl (SVM model)

- tfidf_vectorizer.pkl (TF-IDF transformer)



