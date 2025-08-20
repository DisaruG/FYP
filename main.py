from src.preprocessing import preprocess_signature

def main():
    print("🔹 Signature Verification System - Basic Version")
    print("Step 1: Preprocessing the signature image...\n")

    # Run preprocessing on a test image
    test_image = "data/sample_signature.png"
    preprocess_signature(test_image)

    print("\n✅ Preprocessing complete. You can now show this to your supervisor!")

if __name__ == "__main__":
    main()
