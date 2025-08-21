from tkinter import filedialog, messagebox
from FYP.src.feature_extraction import extract_hog_features
from FYP.src.preprocessing import preprocess_signature


def select_file_and_predict(model):
    """UI function to select a signature, preprocess, and predict."""
    file_path = filedialog.askopenfilename(
        title="Select Signature",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )
    if not file_path:
        return

    # Preprocess image
    processed = preprocess_signature(file_path)

    # Extract HOG features
    features = extract_hog_features(processed).reshape(1, -1)

    # Predict using the trained SVM model
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    # Determine result text
    result_text = "Genuine" if prediction == 1 else "Forged"

    # Show success message with prediction and confidence
    messagebox.showinfo(
        "Prediction Successful",
        f"✅ Signature processed successfully!\n\n"
        f"Result: {result_text}\n"
        f"Confidence: {max(prob)*100:.2f}%"
    )
