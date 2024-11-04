#!/usr/bin/env python
import datetime
import os
from glob import glob
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.layers.experimental import preprocessing
from keras.models import Model
from keras.preprocessing.image import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from constants import (
    BATCH_SIZE,
    CLASS_LIST,
    DATA_FOLDER,
    EPOCHS,
    IMAGE_SIZE,
    WEIGHTS_FILE,
)


def load_data(folder: str, target_size: tuple[int, int]):
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

    for image_path in image_paths:
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image)
        label = image_path.split(os.path.sep)[-2]
        print("LABEL: ", label, end="\n")

        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)


def skinlesNetModel(image_width: int, image_height: int, depth: int, classes: int):
    """
    Create and return the SkinlesNet model architecture.

    Returns:
    tensorflow.keras.models.Sequential: The compiled SkinlesNet model
    """
    input_layer = Input(shape=(image_width, image_height, depth))
    model = Conv2D(filters=32, kernel_size=(3, 30), padding="same")(input_layer)
    model = ReLU()(model)
    model = BatchNormalization(axis=-1)(model)
    model = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(model)
    model = ReLU()(model)
    model = BatchNormalization(axis=-1)(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(rate=0.25)(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(model)
    model = ReLU()(model)
    model = BatchNormalization(axis=-1)(model)
    model = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(model)
    model = ReLU()(model)
    model = BatchNormalization(axis=-1)(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = Dropout(rate=0.25)(model)
    model = Flatten()(model)
    model = Dense(units=512)(model)
    model = ReLU()(model)
    model = BatchNormalization(axis=-1)(model)
    model = Dropout(rate=0.5)(model)
    model = Dense(units=classes)(model)
    output = Activation("sigmoid")(model)
    return Model(input_layer, output)


if __name__ == "__main__":
    X, y = load_data(folder=DATA_FOLDER, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    X = X.astype("float") / 255.0
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    SEED = 666
    np.random.seed(SEED)
    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=SEED
    )

    (X_train, X_valid, y_train, y_valid) = train_test_split(
        X_train, y_train, stratify=y_train, test_size=0.2, random_state=SEED
    )

    model = skinlesNetModel(
        image_width=IMAGE_SIZE,
        image_height=IMAGE_SIZE,
        depth=3,
        classes=len(mlb.classes_),
    )
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )
    result = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print(f"TEST ACCURACY: {result[1]}")
