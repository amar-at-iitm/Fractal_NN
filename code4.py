import numpy as np
import wandb

# Importing from local directory
from functions import load_data, preprocess_data, initialize_network, compute_accuracy
from optimizers import optimizers
from sweep_config import sweep_config
from propagation import forward_propagation, backpropagation



# Training Function
def train():
    wandb.init(entity="amar74384-iit-madras", project="DA6401_assign_1")
    config = wandb.config

    # Generating a meaningful run name using config values
    run_name = f"run_hl-{config.hidden_layers}_bs-{config.batch_size}_act-{config.activation}_opt-{config.optimizer}"
    wandb.run.name = run_name
    wandb.run.save()

    # Loading and preprocessing dataset
    (x_train, y_train), (x_test, y_test) = load_data('fashion-mnist.npz')
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Splitting 10% of training data as validation set
    val_split = int(0.1 * x_train.shape[0])
    x_val, y_val = x_train[:val_split], y_train[:val_split]
    x_train, y_train = x_train[val_split:], y_train[val_split:]

    # Initializing the network
    layer_sizes = [784] + [config.hidden_size] * config.hidden_layers + [10]
    weights, biases = initialize_network(layer_sizes, config.weight_init)

    # Selecting optimizer
    optimizer_class = optimizers.get(config.optimizer, optimizers["sgd"])
    optimizer = optimizer_class(config.learning_rate, weight_decay=config.weight_decay)

    # Training Loop
    for epoch in range(config.epochs):
        indices = np.random.permutation(x_train.shape[0])
        x_train, y_train = x_train[indices], y_train[indices]
        
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0

        # Mini-batch training
        for i in range(0, x_train.shape[0], config.batch_size):
            x_batch = x_train[i:i + config.batch_size]
            y_batch = y_train[i:i + config.batch_size]
            
            activations, z_values = forward_propagation(x_batch, weights, biases, config.activation)
            gradients_w, gradients_b = backpropagation(activations, z_values, weights, y_batch, config.activation, config.weight_decay)
            optimizer.update(weights, biases, gradients_w, gradients_b)
            
            # Computing batch loss with L2 regularization
            batch_loss = -np.mean(np.sum(y_batch * np.log(activations[-1] + 1e-8), axis=1)) + \
                         (config.weight_decay / 2) * sum(np.sum(w**2) for w in weights)
            epoch_loss += batch_loss * x_batch.shape[0]

            correct_predictions += np.sum(np.argmax(activations[-1], axis=1) == np.argmax(y_batch, axis=1))
            total_samples += x_batch.shape[0]

        # Computing Training Loss and Accuracy
        train_activations, _ = forward_propagation(x_train, weights, biases, config.activation)
        #####################################################################################
        # Computing training loss with L2 regularization (Cross-Entropy + L2 Penalty)
        train_loss = -np.mean(np.sum(y_train * np.log(train_activations[-1] + 1e-8), axis=1)) + \
                     (config.weight_decay / 2) * sum(np.sum(w**2) for w in weights)
        #######################################################################################
        train_acc = compute_accuracy(x_train, y_train, weights, biases, config.activation)

        # Compute Validation Loss and Accuracy
        val_activations, _ = forward_propagation(x_val, weights, biases, config.activation)
        #################################################################################
        # Computing validation loss with L2 regularization (Cross-Entropy + L2 Penalty)
        val_loss = -np.mean(np.sum(y_val * np.log(val_activations[-1] + 1e-8), axis=1)) + \
                   (config.weight_decay / 2) * sum(np.sum(w**2) for w in weights)
        #################################################################################
        val_acc = compute_accuracy(x_val, y_val, weights, biases, config.activation)

        # Computing Test Accuracy
        test_acc = compute_accuracy(x_test, y_test, weights, biases, config.activation)

        # Loging results to Weights & Biases
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "Test_Acc": test_acc
        })

    return weights, biases

# Sweeps for hyperparameter tuning
sweep_id = wandb.sweep(sweep_config, project="DA6401_assign_1")
wandb.agent(sweep_id, function=train, count=100)

