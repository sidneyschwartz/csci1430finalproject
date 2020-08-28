# ASSNet Final Project:
Austen Royer, Scott Huson, Sidney Schwartz

CSCI 1430 - James Tompkin

## Requirements
We used Tensorflow for this implementation of the Context-Aware Net
Other libraries include:
 * OpenCV
 * scikit-learn
 * Pillow
 * Matplotlib
 * Keras
 * h5py

## Training
To train the model, run:
``` python code/model.py ```
args to enter:
--batch_size to specify how large the batch_size will be for training
--resize_size default 1024, square dimension to resize training images to
--epochs number of epochs to train for
--validate_size number of validation images to use for validation
--data_subset size of part A of ShanghaiTech to limit training data to, helpful if you dont want to load in tons of files

This will save weights and model structure into an h5 file in the /model directory. 

## Inference
To make an inference on a certain photo, run:
``` python code/inference.py ```
You may need to change the image location inside the file.
It will output the density map prediction and ground truth.

## Analysis
To run analysis on the model, run:
``` python code/analysis.py ```

