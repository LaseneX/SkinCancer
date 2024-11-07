import os

CURRENT_FOLDER = os.path.dirname(__file__)
IMAGE_SIZE = 224
DATA_FOLDER = r"C:\Users\Lasene\Downloads\SkinCancerDataset\skin-lesions"
""" The classes are:

    | Label | Description |
    |:-----:|-------------|
    |   0   | bcc         |
    |   1   | melanoma    |
"""
CLASS_LIST = ["bcc", "mel"]
BATCH_SIZE = 16
EPOCHS = 150
SAVE_FOLDER = os.path.join(CURRENT_FOLDER, "checkpointer")
WEIGHTS_FILE = os.path.join(SAVE_FOLDER, "checkpointer/model_v.0.0.1.weights.h5")

CLASS_ITEMS = {items: i for i, items in enumerate(CLASS_LIST)}