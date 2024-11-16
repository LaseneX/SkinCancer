# Overview

This project is an image classification model designed to classify images into specific categories using a deep learning approach. Built and trained using the Keras API, the model aims to accurately identify skin cancer types based on provided image data. Currently, the model supports two classes: `bcc` (Basal Cell Carcinoma) and `melanoma`. 

This is a hobby project created to explore and understand how deep learning models work. The code is structured in a way that allows users to easily retrain the model with additional classes if needed.

---

## Project Structure

The codebase is organized as follows:

- **`train.py`**: Script for training the model. No modification is required to train on a custom dataset.
- **`inference.py`**: Script to load the trained model and make predictions on images.
- **`api.py`**: Provides a web API interface for the model using FastAPI, enabling RESTful access.
- **`constants.py`**: Contains constants and configuration variables, such as batch size, epochs, file paths, and class names. For custom training, users must modify this file as needed.
- **`download.py`**: Script to download a pre-trained model from cloud storage, allowing users with limited resources to test the model without retraining.

---

## Usage

### Training the Model

To train the model, ensure the dataset path specified in `constants.py` is correct and update the class names accordingly. Then, run the following command:

```bash
python train.py
```

This script will:
- Load the dataset.
- Preprocess the images.
- Train the model.
- Save the training progress as a PNG file with a timestamped filename.
- Save the model weights to the path specified in the `WEIGHTS_FILE` variable within `constants.py`.

### Making Predictions Locally

To make predictions on a new image locally without using the REST API, run:

```bash
python inference.py
```

Ensure the testing image path is set correctly in the `constants.py` file or directly in `inference.py`.

### Using the Web API

To set up the API, run:

```bash
python api.py
```

The API provides two endpoints:
1. **`/health`**: For server health checks.
2. **`/predict`**: For making predictions. Users must send an image file as part of the request.

### Download prebuilt model

To download prebuilt model run :
```bash
python download.py `version`
```
where version may be `v0.1` or `v0.2`

---

## Results

- **Training Metrics**: Accuracy and loss metrics are saved as PNG images with timestamped filenames (e.g., `1730888615.0278668_plot.png`). These plots help visualize the model’s performance during training.
- **Prediction Performance**: The model achieves an accuracy exceeding 95% for the two supported classes (`bcc` and `melanoma`).

---

## Contributions

Contributions are welcome! Whether it’s enhancing the source code or improving the documentation, your input is greatly appreciated.

---