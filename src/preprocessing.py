import cv2

def preprocess_signature(image_path, resize_dim=(128, 64)):
    """
    Convert signature image to grayscale, threshold, and resize for HOG features.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    processed = cv2.resize(thresh, resize_dim)
    return processed
