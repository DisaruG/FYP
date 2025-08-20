import cv2
import os
import matplotlib.pyplot as plt

def preprocess_signature_for_hog(image_path,
                                 output_folder="output/processed_signatures",
                                 resize_dim=(128, 64),  # Height x Width for HOG
                                 show_preview=True):
    """
    Preprocess a signature image for HOG + SVM classification.
    Steps:
        1. Grayscale conversion
        2. Noise removal (median blur)
        3. Adaptive thresholding
        4. Resize to standard dimensions
    Returns:
        preprocessed image ready for HOG feature extraction
    """

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove small noise
    denoised = cv2.medianBlur(gray, 3)

    # Adaptive thresholding (better for varied lighting)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Resize to standard dimensions (for HOG)
    processed = cv2.resize(thresh, resize_dim)

    # Save processed image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_processed.png"), processed)

    # Optional preview
    if show_preview:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original")
        axs[0].axis("off")

        axs[1].imshow(thresh, cmap='gray')
        axs[1].set_title("Thresholded + Denoised")
        axs[1].axis("off")

        axs[2].imshow(processed, cmap='gray')
        axs[2].set_title("Resized for HOG")
        axs[2].axis("off")

        plt.suptitle(f"Processing: {os.path.basename(image_path)}")
        plt.show()

    return processed


def preprocess_multiple_for_hog(folder_path, output_folder="output/processed_signatures"):
    """
    Preprocess all images in a folder for HOG + SVM.
    Returns a list of preprocessed images.
    """
    processed_images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            processed = preprocess_signature_for_hog(img_path, output_folder, show_preview=False)
            processed_images.append(processed)
    return processed_images
