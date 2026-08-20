import numpy as np
import wandb
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")  # Set non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

# Importing from local directory
from functions import load_data, preprocess_data, initialize_network, compute_accuracy
from optimizers import optimizers
from propagation import forward_propagation, backpropagation
from best_run import best_run_config  # Importing best run config instead of sweep_config


def log_confusion_matrix(x_test, y_test, model_weights, model_biases, activation, title="Confusion Matrix"):
    activations, _ = forward_propagation(x_test, model_weights, model_biases, activation)
    y_pred = np.argmax(activations[-1], axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[str(i) for i in range(10)], 
                yticklabels=[str(i) for i in range(10)])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)

    wandb.log({title: wandb.Image(plt)})
    plt.close()


def train():
    wandb.init(entity="amar74384-iit-madras", project="DA6401_assign_1", config=best_run_config)  # Using best run config
    
    config = wandb.config

    run_name = f"load_confusion_matrix"
    wandb.run.name = run_name
    wandb.run.save()
    
    
    (x_train, y_train), (x_test, y_test) = load_data('fashion-mnist.npz')
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
    
    val_split = int(0.1 * x_train.shape[0])
    x_val, y_val = x_train[:val_split], y_train[:val_split]
    x_train, y_train = x_train[val_split:], y_train[val_split:]

    layer_sizes = [784] + [config.hidden_size] * config.hidden_layers + [10]
    weights, biases = initialize_network(layer_sizes, config.weight_init)

    optimizer_class = optimizers.get(config.optimizer, optimizers["sgd"])
    optimizer = optimizer_class(config.learning_rate, weight_decay=config.weight_decay)

    for epoch in range(config.epochs):
        indices = np.random.permutation(x_train.shape[0])
        x_train, y_train = x_train[indices], y_train[indices]
        
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for i in range(0, x_train.shape[0], config.batch_size):
            x_batch = x_train[i:i + config.batch_size]
            y_batch = y_train[i:i + config.batch_size]
            
            activations, z_values = forward_propagation(x_batch, weights, biases, config.activation)
            gradients_w, gradients_b = backpropagation(activations, z_values, weights, y_batch, config.activation, config.weight_decay)
            optimizer.update(weights, biases, gradients_w, gradients_b)
            
            batch_loss = -np.mean(np.sum(y_batch * np.log(activations[-1] + 1e-8), axis=1))
            epoch_loss += batch_loss * x_batch.shape[0]
            correct_predictions += np.sum(np.argmax(activations[-1], axis=1) == np.argmax(y_batch, axis=1))
            total_samples += x_batch.shape[0]
        
        train_loss = -np.mean(np.sum(y_train * np.log(forward_propagation(x_train, weights, biases, config.activation)[0][-1] + 1e-8), axis=1)) + \
                     (config.weight_decay / 2) * sum(np.sum(w**2) for w in weights)
        train_acc = compute_accuracy(x_train, y_train, weights, biases, config.activation)

        val_loss = -np.mean(np.sum(y_val * np.log(forward_propagation(x_val, weights, biases, config.activation)[0][-1] + 1e-8), axis=1)) + \
                   (config.weight_decay / 2) * sum(np.sum(w**2) for w in weights)
        val_acc = compute_accuracy(x_val, y_val, weights, biases, config.activation)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        log_confusion_matrix(x_test, y_test, weights, biases, config.activation)
        
    return weights, biases

train() 
