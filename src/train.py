#!/usr/bin/env python
import datetime
import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential

from constants import (
    BATCH_SIZE,
    CLASS_LIST,
    DATA_FOLDER,
    EPOCHS,
    IMAGE_SIZE,
    WEIGHTS_FILE,
)


def load_data(folder: str, classes: list):
    """
    Load the images from the folder and return the images and labels.

    Args:
    folder (str): Path to the data folder
    classes (list): List of class names

    Returns:
    list: A list of [image, label] pairs
    """

    images_grid = []
    for cls in classes:
        path = os.path.join(folder, cls)
        label = classes.index(cls)

        for img in os.listdir(path):
            if img.endswith(".jpg"):
                img_path = os.path.join(path, img)
                print(f"Reading the {img_path}")
                img_array = cv2.imread(img_path)
                img_array_resized = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))

                images_grid.append([img_array_resized, label])

    return images_grid


def create_train_test(images_grid: list):
    """
    Create train and test datasets from the loaded images.

    Args:
    images_grid (list): List of [image, label] pairs

    Returns:
    tuple: (X_train, X_test, y_train, y_test) - Train and test datasets
    """
    X = []
    y = []
    for features, labels in images_grid:
        X.append(features)
        y.append(labels)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return (X_train, X_test, y_train, y_test)


def skinlesNetModel():
    """
    Create and return the SkinlesNet model architecture.

    Returns:
    tensorflow.keras.models.Sequential: The compiled SkinlesNet model
    """
    model = Sequential(
        [
            preprocessing.Rescaling(scale=1.0 / 255),
            preprocessing.RandomFlip("horizontal_and_vertical"),
            preprocessing.RandomZoom(
                height_factor=(-0.05, -0.15), width_factor=(-0.05, -0.15)
            ),
        ]
    )

    # 1st Convolutional Input Layer
    model.add(
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    )
    model.add(MaxPooling2D((2, 2)))

    # 2nd Convolutional Input Layer
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    # 3rd Convolutional Layer
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    # 4th Convolutional Layer
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.5))

    # Flatten Layer
    model.add(Flatten())

    # Hidden Layer
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(3, activation="softmax"))

    return model


if __name__ == "__main__":
    # Load the data
    images_grid = load_data(DATA_FOLDER, CLASS_LIST)
    print("[+] Data loaded")

    # Create train and test datasets
    X_train, X_test, y_train, y_test = create_train_test(images_grid)
    print("[+] Train Test Split created")

    # Create and compile the model
    model = skinlesNetModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, ema_momentum=0.99)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    print("[+] Model Compiled")

    # Set up TensorBoard callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    # Train the model
    print("[+] Training Model")
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=[tensorboard_callback],
    )

    # Save the model weights
    model.save_weights(WEIGHTS_FILE)
    print("[+] Model weights saved")