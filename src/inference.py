from constants import CLASS_LIST, IMAGE_SIZE, WEIGHTS_FILE
from train import contruct_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


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
    prediction = model.predict_classes(preprocessed_image)
    predicted_class = CLASS_LIST[prediction[0]]

    print(f"Predicted class: {predicted_class}")
    return predicted_class