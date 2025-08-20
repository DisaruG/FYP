import cv2
import matplotlib.pyplot as plt

def preprocess_signature(image_path):
    #Load the image
    img = cv2.imread(image_path)

    #convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Apply thresholding (black and white)
    _, thresh = cv2.threshold(gray,127, 255, cv2.THRESH_BINARY_INV)

    #Show Results
    plt.subplots(1, 3)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.subplots(1, 3)
    plt.title("Grayscale")
    plt.imshow(gray, cmap='gray')

    plt.subplots(1, 3)
    plt.title("Thresholded")
    plt.imshow(thresh, cmap='gray')

    plt.show()