import numpy as np 
# from tensorflow.keras.datasets import fashion_mnist
from src.propagation import forward_propagation


def load_data(filepath):
    with np.load(filepath) as data:
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']
    return (x_train, y_train), (x_test, y_test)


# def load_data(filepath):
#     (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data() # using directly from keras
#     with np.load(filepath) as data:
#         x_train, y_train = data['x_train'], data['y_train']
#         x_test, y_test = data['x_test'], data['y_test']
#     return (x_train, y_train), (x_test, y_test)

def preprocess_data(x, y):
    """Normalize and reshape input, one-hot encode labels."""
    x = x.reshape(x.shape[0], -1) / 255.0  # Flatten and normalize
    y_one_hot = np.eye(10)[y]  # Convert to one-hot encoding
    return x, y_one_hot

# Initializing The Network
def initialize_network(layer_sizes, init_method="random"):
    """
    Initializes network weights and biases.

    Args:
        layer_sizes: List specifying the number of neurons in each layer
        init_method: Weight initialization method ("xavier" or "random")

    Returns:
        weights: List of weight matrices
        biases: List of bias vectors
    """
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        if init_method == "xavier":
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
        else:  # Default random initialization
            limit = 0.1
        weights.append(np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1])))
        biases.append(np.zeros((1, layer_sizes[i + 1])))

    return weights, biases

# Accuracy Computation
def compute_accuracy(x, y, weights, biases, activation):
    """
    Computes accuracy of the model.

    Args:
        x: Input data
        y: True labels (one-hot encoded)
        weights: List of weight matrices
        biases: List of bias vectors
        activation: Activation function used in hidden layers

    Returns:
        Accuracy score
    """
    activations, _ = forward_propagation(x, weights, biases, activation)
    predictions = np.argmax(activations[-1], axis=1)
    y_labels = np.argmax(y, axis=1)
    return np.mean(predictions == y_labels)
