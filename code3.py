import numpy as np

# Importing from local directory
from functions import load_data, preprocess_data, initialize_network, compute_accuracy
from optimizers import optimizers
from propagation import forward_propagation, backpropagation


# Training Function
def train_network(x_train, y_train, x_val, y_val, layer_sizes, optimizer_name, epochs=10, batch_size=32, learning_rate=0.01):
    weights, biases = initialize_network(layer_sizes)
    optimizer = optimizers[optimizer_name](learning_rate)
    
    for epoch in range(epochs):
        indices = np.random.permutation(x_train.shape[0])
        x_train, y_train = x_train[indices], y_train[indices]
        
        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            activations, z_values = forward_propagation(x_batch, weights, biases, 'relu')
            gradients_w, gradients_b = backpropagation(activations, z_values, weights, y_batch, 'relu', 0.0)
            optimizer.update(weights, biases, gradients_w, gradients_b)
        
        train_activations, _ = forward_propagation(x_train, weights, biases, 'relu')
        loss = -np.mean(np.sum(y_train * np.log(train_activations[-1] + 1e-8), axis=1))  # Cross-Entropy Loss

        val_acc = compute_accuracy(x_val, y_val, weights, biases, 'relu')
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Val Acc: {val_acc:.4f}")
    
    return weights, biases

# Loading and Preprocessing Fashion-MNIST Dataset
(x_train, y_train), (x_test, y_test) = load_data('fashion-mnist.npz')
x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

# Ask User for Optimizer
while True:
    optimizer_name = input(f"Select an optimizer {list(optimizers.keys())}: ").strip().lower()
    if optimizer_name in optimizers:
        break
    print("Invalid choice. Please select from the available options.")

# Train the Network
train_network(x_train, y_train, x_test, y_test, [784, 128, 64, 10], optimizer_name)
