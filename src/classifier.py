import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from FYP.src.preprocessing import preprocess_signature
from FYP.src.feature_extraction import extract_hog_features

def load_dataset(genuine_folder, forged_folder):
    """
    Load dataset images and extract HOG features.
    Genuine = 1, Forged = 0
    """
    X, y = [], []
    for folder, label in [(genuine_folder, 1), (forged_folder, 0)]:
        for f in os.listdir(folder):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                img = preprocess_signature(os.path.join(folder, f))
                X.append(extract_hog_features(img))
                y.append(label)
    return np.array(X), np.array(y)

def train_svm(X, y):
    """
    Train a linear SVM classifier.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = svm.SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Demo Accuracy: {acc*100:.2f}%")
    return model

def save_model(model, filename="svm_model.joblib"):
    joblib.dump(model, filename)
    print(f"💾 Model saved to {filename}")

def load_model(filename="svm_model.joblib"):
    return joblib.load(filename)

def predict(model, features):
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    result_text = "Genuine" if prediction == 1 else "Forged"
    return result_text, prob
