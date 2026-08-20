# This program loads the mnist dataset to the local directory 
import tensorflow as tf
import numpy as np
import os


# Loading MNIST dataset
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("MNIST dataset loaded successfully!")
print(f"Training data shape: {x_train.shape}, Labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}, Labels shape: {y_test.shape}")

# Defining save path
save_path = "mnist.npz"

# Saving dataset as a compressed NumPy archive
np.savez_compressed(save_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

print(f"Dataset saved locally as '{save_path}' in {os.getcwd()}")
