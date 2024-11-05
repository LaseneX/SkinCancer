#!/usr/bin/env python
import os
from glob import glob
from time import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img, to_categorical
from numpy._typing import NDArray
from sklearn.model_selection import train_test_split

from constants import (
    BATCH_SIZE,
    CLASS_ITEMS,
    CLASS_LIST,
    DATA_FOLDER,
    EPOCHS,
    IMAGE_SIZE,
    WEIGHTS_FILE,
)


def load_data(
    folder: str, target_size: tuple[int, int]
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Load the images from the folder and return the images and labels.

    Args:
    folder (str): Path to the data folder
    classes (list): List of class names

    Returns:
    list: A list of [image, label] pairs
    """
    images = []
    labels = []
    images_path_pattern = os.path.join(folder, "*/*.jpg")
    image_paths = glob(images_path_pattern)
    print(CLASS_ITEMS)

    for image_path in image_paths:
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image)
        label = image_path.split(os.path.sep)[-2]
        images.append(image)
        labels.append(CLASS_ITEMS[label])

    print("NUMBER of Images: ", len(images))
    print("NUMBER of labels: ", len(labels))

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.3, random_state=42
    )
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print("y_train: ", len(y_train))
    print("y_test: ", len(y_test))

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def normalize_pixel_values(
    X_train: NDArray, X_test: NDArray
) -> Tuple[NDArray, NDArray]:
    """Add documentation here"""
    X_train_norm = X_train.astype("float32")
    X_test_norm = X_test.astype("float32")

    X_train_norm = X_train_norm / 255.0
    X_test_norm = X_test_norm / 255.0

    return X_train_norm, X_test_norm


def contruct_model():
    model = Sequential()
    model.add(
        Conv2D(
            IMAGE_SIZE,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        )
    )

    model.add(
        Conv2D(
            IMAGE_SIZE,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
        )
    )

    model.add(MaxPooling2D((2, 2)))
    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
        )
    )
    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
        )
    )

    model.add(MaxPooling2D((2, 2)))

    model.add(
        Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
        )
    )
    model.add(
        Conv2D(
            128,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            padding="same",
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(len(CLASS_LIST), activation="softmax"))

    optimizer = SGD(learning_rate=0.0001, momentum=0.9)

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def summarize_diagnostics(training_history):
    plt.subplot(211)
    plt.title("Cross Entropy Loss")
    plt.plot(training_history.history["loss"], color="blue", label="train")
    plt.plot(training_history.history["val_loss"], color="orange", label="test")

    plt.subplot(212)
    plt.title("Classification Accuracy")
    plt.plot(training_history.history["acc"], color="blue", label="train")
    plt.plot(training_history.history["val_acc"], color="orange", label="test")
    filename = f"{time()}_plot.png"
    plt.savefig(filename)
    plt.close()


def run_test():
    x_train, y_train, x_test, y_test = load_data(DATA_FOLDER, (IMAGE_SIZE, IMAGE_SIZE))
    x_train, x_test = normalize_pixel_values(x_train, x_test)
    print("X_train: ", len(x_train))
    print("Y_train: ", len(y_train))
    print("X_test: ", len(x_test))
    print("Y_test: ", len(y_test))

    model = contruct_model()
    datagen = ImageDataGenerator(
        width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
    )
    iterator_train = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
    print("SHAPE: ", x_train.shape)
    steps = int(x_train.shape[0] / 64)
    history = model.fit(
        iterator_train,
        steps_per_epoch=steps,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
    )
    _, acc = model.evaluate(x_test, y_test)
    print("%.3f" % (acc * 100))
    summarize_diagnostics(history)


if __name__ == "__main__":
    run_test()
