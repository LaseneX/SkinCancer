from constants import CLASS_LIST, IMAGE_SIZE, WEIGHTS_FILE, TESTING_IMAGE
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np


def load_image(image_path):
    """
    Preprocess the input image for model prediction.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        numpy.ndarray: Preprocessed image array.
    """
    image = load_img(image_path, target_size = (IMAGE_SIZE, IMAGE_SIZE))
    image = img_to_array(image)
    image = image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
    image = image.astype("float32")
    image = image/255.0
    return image


def predict_image(image_path):
    """
    Load the trained model weights, preprocess the input image, and make a prediction.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        str: Predicted class name.
    """
    model = load_model(WEIGHTS_FILE)

    preprocessed_image = load_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class = CLASS_LIST[np.argmax(prediction)]
    predicted_class_prob = np.max(prediction)

    print(f"Predicted class: {predicted_class}")
    print(f"Predicted probality: {predicted_class_prob}")
    return predicted_class, predicted_class_prob

if __name__ == "__main__":
    predict_image(TESTING_IMAGE)