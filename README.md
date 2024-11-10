# Overview

This project is an image classification model that classifies images into specific categories using machine learning. The model was built and trained using Keras and TensorFlow, with the aim of accurately identifying skin cancer classes based on provided image data.

## Project Structure

train.py: Script for training the model.
inference.py: Script to load the trained model and make predictions on images.
api.py: API interface for the model to be accessed using FastAPI.
constants.py: Stores constants and configuration variables, such as batch size, epochs, and file paths.
testing/: Folder for test images.
requirements.txt: Lists the Python dependencies.
1730888615.0278668_plot.png and 1731073056.6615644_plot.png: Sample training plots showing model loss and accuracy over epochs.
.gitignore: Specifies files and folders to ignore in Git tracking.

## Usage
### Training the Model
To train the model, run the following command: 
```bash 
$ python train.py 
```

This will load the dataset, preprocess the images, and train the model. Training progress will be plotted and saved in a PNG file with a timestamped filename. Model weights will be saved to the location specified by WEIGHTS_FILE in constants.py.

To make predictions on a new image, run:  
```bash 
$ python inference.py
```

You can set the testing image to what you desire at constants.py file.

The api.py script allows the model to be accessed via an API endpoint. Set up the API by running: 
```bash
$ python api.py
```

## Results

Training metrics (accuracy and loss) are saved as images with timestamped filenames, such as 1730888615.0278668_plot.png. These plots provide insight into the model’s performance during training.