import tkinter as tk
from src.classifier import load_dataset, train_svm
from src.ui import select_file_and_predict

def main():
    genuine_folder = "data/genuine"
    forged_folder = "data/forged"

    # Load dataset and train SVM
    X, y = load_dataset(genuine_folder, forged_folder)
    model = train_svm(X, y)

    # Simple Tkinter UI
    root = tk.Tk()
    root.title("Offline Signature Verification (Demo)")
    root.geometry("400x150")

    tk.Label(root, text="Offline Signature Verification (Demo)", font=("Arial", 14)).pack(pady=10)
    tk.Button(root, text="Select Signature to Verify", font=("Arial", 12),
              command=lambda: select_file_and_predict(model)).pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
