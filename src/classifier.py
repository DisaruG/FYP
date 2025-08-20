import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.preprocessing import preprocess_signature
from src.feature_extraction import extract_hog_features

def load_dataset(genuine_folder, forged_folder):
    """Load a small dataset and extract HOG features."""
    X, y = [], []

    for folder, label in [(genuine_folder, 1), (forged_folder, 0)]:
        for f in os.listdir(folder):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                img = preprocess_signature(os.path.join(folder, f))
                X.append(extract_hog_features(img))
                y.append(label)
    return np.array(X), np.array(y)

def train_svm(X, y):
    """Train an SVM on HOG features."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = svm.SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Demo Accuracy: {acc*100:.2f}%")
    return model
