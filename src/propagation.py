import numpy as np
from activation import activation_fn, activation_derivative  

def forward_propagation(x, weights, biases, activation):
    """
    Performs forward propagation through the neural network.

    Args:
        x: Input data (num_samples, input_size)
        weights: List of weight matrices for each layer
        biases: List of bias vectors for each layer
        activation: Activation function for hidden layers ("relu", "sigmoid", "tanh")

    Returns:
        activations: List of layer activations including input and output
        z_values: List of pre-activation values (z) for each layer
    """
    activations = [x]
    z_values = []

    for i in range(len(weights) - 1):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        z_values.append(z)
        activations.append(activation_fn(z, activation))

    # Output layer uses softmax activation
    z = np.dot(activations[-1], weights[-1]) + biases[-1]
    z_values.append(z)
    activations.append(activation_fn(z, "softmax"))

    return activations, z_values

def backpropagation(activations, z_values, weights, y_true, activation, weight_decay):
    """
    Performs backward propagation through the network with L2 regularization.

    Args:
        activations: List of layer activations
        z_values: List of pre-activation values
        weights: List of weight matrices
        y_true: True labels (one-hot encoded)
        activation: Activation function for hidden layers ("relu", "sigmoid", "tanh")
        weight_decay: L2 regularization hyperparameter (lambda)

    Returns:
        gradients_w: List of gradients for weights (with L2 regularization)
        gradients_b: List of gradients for biases
    """
    num_layers = len(weights)
    m = y_true.shape[0]

    gradients_w = []
    gradients_b = []

    # Compute output layer gradient (Softmax + Cross-Entropy)
    y_pred = activation_fn(z_values[-1], "softmax")  
    delta = (y_pred - y_true)  # ∂L/∂z_L

    # Compute gradients for output layer (with L2 regularization)
    h_prev = activations[-2] if num_layers > 1 else activations[0]
    gradients_w.append((np.dot(h_prev.T, delta) / m) + (weight_decay * weights[-1]))  # L2 added

    #gradients_w.append((np.dot(delta.T, h_prev) / m) + (weight_decay * weights[-1]))  # L2 added
    gradients_b.append(np.sum(delta, axis=0, keepdims=True) / m)

    # Backpropagate through hidden layers
    for l in reversed(range(num_layers - 1)):
        W_next = weights[l + 1]
        delta = np.dot(delta, W_next.T) * activation_derivative(z_values[l], activation)

        h_prev = activations[l] if l > 0 else activations[0]
        gradients_w.append((np.dot(h_prev.T, delta) / m) + (weight_decay * weights[l]))  # L2 added

        #gradients_w.append((np.dot(delta.T, h_prev) / m) + (weight_decay * weights[l]))  # L2 added
        gradients_b.append(np.sum(delta, axis=0, keepdims=True) / m)

    # Reverse lists to match weight order
    gradients_w.reverse()
    gradients_b.reverse()

    return gradients_w, gradients_b
