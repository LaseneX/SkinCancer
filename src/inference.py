import cv2
import numpy as np

from constants import CLASS_LIST, IMAGE_SIZE, WEIGHTS_FILE
from train import skinlesNetModel


def preprocess_image(image_path):
    """
    Preprocess the input image for model prediction.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        numpy.ndarray: Preprocessed image array.
    """
    img_array = cv2.imread(image_path)
    img_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def predict_image(image_path):
    """
    Load the trained model, preprocess the input image, and make a prediction.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        str: Predicted class name.
    """
    model = skinlesNetModel()
    model.load_weights(WEIGHTS_FILE)

    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)

    predicted_class_index = np.argmax(prediction)
    predicted_class = CLASS_LIST[predicted_class_index]

    print(f"Predicted class: {predicted_class}")
    return predicted_class