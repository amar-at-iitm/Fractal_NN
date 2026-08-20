import numpy as np
from alpha_fractal_function import alpha_fractalize_second_derivative, alpha_fractalize_first_derivative, alpha_fractalize

def activation_fn(z, fn):
    """Applies the activation function."""
    if fn.lower() == "relu":  # To handle case-insensitive "ReLU"
        return np.maximum(0, z)
    elif fn == "sigmoid":
        return 1 / (1 + np.exp(-z))
    elif fn == "tanh":
        return np.tanh(z)
    elif fn == "softmax":
        return softmax(z)  
    else:
        raise ValueError("Unsupported activation function")
    

def activation_derivative(z, fn):
    """Computes the derivative of the activation function."""
    if fn.lower() == "relu":  # Updated to handle case-insensitive "ReLU"
        return (z > 0).astype(float)
    elif fn == "sigmoid":
        sig = 1 / (1 + np.exp(-z))
        return sig * (1 - sig)
    elif fn == "tanh":
        return 1 - np.tanh(z) ** 2
    else:
        raise ValueError("Unsupported activation function")


def softmax(z):
    """Applies the softmax function."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability trick
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
