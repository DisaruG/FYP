import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

# Import your existing modules
from src.classifier import load_model, predict
from src.preprocessing import preprocess_signature
from src.feature_extraction import extract_hog_features

# ---------- CONFIG ----------
MODEL_PATH = "svm_model.joblib"
IMAGE_DISPLAY_SIZE = (300, 150)
# ----------------------------

def run_gui(model):
    """
    Tkinter GUI for selecting a signature and predicting Genuine/Forged.
    """
    def browse_image():
        filepath = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if not filepath:
            return

        # Display the image
        img = Image.open(filepath)
        img.thumbnail(IMAGE_DISPLAY_SIZE)
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Predict using the trained model
        try:
            processed = preprocess_signature(filepath)
            features = extract_hog_features(processed).reshape(1, -1)
            result, prob = predict(model, features)
            result_text.set(f"Prediction: {result}\nConfidence: {max(prob)*100:.2f}%")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    # Create GUI window
    root = tk.Tk()
    root.title("Signature Verification Demo")

    # Browse button
    browse_btn = tk.Button(root, text="Select Signature Image", command=browse_image)
    browse_btn.pack(pady=10)

    # Image display
    image_label = tk.Label(root)
    image_label.pack()

    # Result label
    result_text = tk.StringVar()
    result_label = tk.Label(root, textvariable=result_text, font=("Arial", 14))
    result_label.pack(pady=10)

    root.mainloop()

def main():
    """
    Load model and open GUI.
    """
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("Error", f"Model file '{MODEL_PATH}' not found!")
        return

    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded ✅")

    run_gui(model)

if __name__ == "__main__":
    main()
