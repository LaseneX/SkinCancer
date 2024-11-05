IMAGE_SIZE = 224
DATA_FOLDER = "../../skin_cancer_dataset/"
""" The classes are:

    | Label | Description |
    |:-----:|-------------|
    |   0   | bcc         |
    |   1   | melanoma    |
"""
CLASS_LIST = ["bcc", "melanoma"]
BATCH_SIZE = 4
EPOCHS = 50
WEIGHTS_FILE = "checkpointer/model_v.0.0.1.weights.h5"

CLASS_ITEMS = {items: i for i, items in enumerate(CLASS_LIST)}
