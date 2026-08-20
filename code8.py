import numpy as np
import wandb

# Importing from local directory
from optimizers import optimizers
from sweep_config import sweep_config  
from propagation import forward_propagation, backpropagation
from functions import load_data, preprocess_data, initialize_network

# Compute Loss Function
def compute_loss(y_true, y_pred, loss_type):
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)  # Prevent log(0) issues
    if loss_type == "cross_entropy":
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    elif loss_type == "squared_error":
        return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))
    else:
        raise ValueError("Invalid loss type")

# Training Function
def train():
    wandb.init(entity="amar74384-iit-madras", project="DA6401_assign_1") 
    
    config = wandb.config
    run_name = f"hl-{config.hidden_layers}_bs-{config.batch_size}_act-{config.activation}_opt-{config.optimizer}"
    wandb.run.name = run_name
    wandb.run.save()
    
    (x_train, y_train), (x_test, y_test) = load_data('fashion-mnist.npz')
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
    
    # Train-validation split
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
        
        for i in range(0, x_train.shape[0], config.batch_size):
            x_batch, y_batch = x_train[i:i + config.batch_size], y_train[i:i + config.batch_size]
            activations, caches = forward_propagation(x_batch, weights, biases, config.activation)
            gradients_w, gradients_b = backpropagation(activations, caches, weights, y_batch, config.activation, config.weight_decay)
            optimizer.update(weights, biases, gradients_w, gradients_b)
        
        # Compute losses only once per epoch
        train_pred = forward_propagation(x_train, weights, biases, config.activation)[0][-1]
        val_pred = forward_propagation(x_val, weights, biases, config.activation)[0][-1]
        
        train_loss_ce = compute_loss(y_train, train_pred, "cross_entropy")
        val_loss_ce = compute_loss(y_val, val_pred, "cross_entropy")

        
        train_loss_se = compute_loss(y_train, train_pred, "squared_error")
        val_loss_se = compute_loss(y_val, val_pred, "squared_error")
        
        wandb.log({
            "Train_Loss_CE": train_loss_ce, 
            "Val_Loss_CE": val_loss_ce,
            "Train_Loss_SE": train_loss_se, 
            "Val_Loss_SE": val_loss_se
        })
        
        
    return weights, biases

# Run W&B sweep
sweep_id = wandb.sweep(sweep_config, project="DA6401_assign_1")
wandb.agent(sweep_id, function=train, count=50)
