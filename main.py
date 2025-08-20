import os
from src.preprocessing import preprocess_multiple_for_hog

def main():
    # Path to your test signatures folder
    test_folder = "data/test_signatures"
    output_folder = "output/processed_signatures"

    # Check if test folder exists
    if not os.path.exists(test_folder):
        print(f"Error: Test folder '{test_folder}' does not exist.")
        return

    # Process images
    processed_images = preprocess_multiple_for_hog(test_folder, output_folder)

    # Print success message
    print(f"✅ Preprocessing completed successfully! {len(processed_images)} images saved to '{output_folder}'.")

if __name__ == "__main__":
    main()
