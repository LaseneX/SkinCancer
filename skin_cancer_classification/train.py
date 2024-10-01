#!/usr/bin/env python
import datetime
import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model

from constants import BATCH_SIZE, CLASS_LIST, DATA_FOLDER, EPOCHS, IMAGE_SIZE


def load_data(folder: str, classes: list):

    images_grid = []
    for cls in classes:
        path = os.path.join(folder, cls)
        label = classes.index(cls)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            print(f"Reading the {img_path}")
            img_array = cv2.imread(img_path)
            img_array_resized = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
            images_grid.append([img_array_resized, label])

    return images_grid


def create_train_test(images_grid):
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
    model = Sequential()

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
    images_grid = load_data(DATA_FOLDER, CLASS_LIST)
    X_train, X_test, y_train, y_test = create_train_test(images_grid)
    model = skinlesNetModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, ema_momentum=0.99)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=[tensorboard_callback],
    )

    # Print the history here
    model.save_weights("./checkpointer/model_v.0.0.1")
