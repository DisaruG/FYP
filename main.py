import os
from src.classifier import load_dataset, train_svm, save_model, load_model, predict
from src.preprocessing import preprocess_signature
from src.feature_extraction import extract_hog_features

def main():
    genuine_folder = "data/genuine"
    forged_folder = "data/forged"

    # 1️⃣ Load dataset and train SVM
    print("Training model on dataset...")
    X, y = load_dataset(genuine_folder, forged_folder)
    model = train_svm(X, y)

    # Save model for later use
    save_model(model)

    # 2️⃣ Demo single signature prediction
    print("\nTesting single signature...")
    test_img = os.path.join(genuine_folder, os.listdir(genuine_folder)[0])
    processed = preprocess_signature(test_img)
    features = extract_hog_features(processed).reshape(1, -1)
    result, prob = predict(model, features)
    print(f"{test_img} -> {result}, Confidence: {max(prob)*100:.2f}%")

    # 3️⃣ Demo batch signature prediction
    print("\nBatch testing genuine folder...")
    for f in os.listdir(genuine_folder):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(genuine_folder, f)
            processed = preprocess_signature(img_path)
            features = extract_hog_features(processed).reshape(1, -1)
            result, prob = predict(model, features)
            print(f"{f:20} | {result:10} | Confidence: {max(prob)*100:.2f}%")

    print("\nBatch testing forged folder...")
    for f in os.listdir(forged_folder):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(forged_folder, f)
            processed = preprocess_signature(img_path)
            features = extract_hog_features(processed).reshape(1, -1)
            result, prob = predict(model, features)
            print(f"{f:20} | {result:10} | Confidence: {max(prob)*100:.2f}%")

if __name__ == "__main__":
    main()
