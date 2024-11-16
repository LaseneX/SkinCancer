import os

CURRENT_FOLDER = os.path.dirname(__file__)
IMAGE_SIZE = 224
DATA_FOLDER = r"C:\Users\Lasene\Downloads\dataset"
""" The classes are:

    | Label | Description |
    |:-----:|-------------|
    |   0   | bcc         |
    |   1   | melanoma    |
"""
CLASS_LIST = ["bcc", "melanoma"]
BATCH_SIZE = 16
EPOCHS = 100
SAVE_FOLDER = os.path.join(CURRENT_FOLDER, "checkpointer")
TESTING_IMAGE = os.path.join(CURRENT_FOLDER, "testing/mel1.jpg")
WEIGHTS_FILE = os.path.join(SAVE_FOLDER, "checkpointer/model_v0.2.weights.h5")
VERSION_URLS = {
    "v0.1": "https://drive.google.com/uc?id=1v5shovs-FnAgmjj4OQ7FJY5qxqFHPPyy&authuser=1",
    "v0.2": "your_file_id_for_v0.2",
    "v0.3": "your_file_id_for_v0.3",
    # Add more versions as needed
}

CLASS_ITEMS = {items: i for i, items in enumerate(CLASS_LIST)}