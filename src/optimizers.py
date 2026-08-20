import numpy as np

# Optimizers
class Optimizer:
    def update(self, weights, biases, gradients_w, gradients_b):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.lr = learning_rate
        self.weight_decay = weight_decay  # L2 Regularization term
    
    def update(self, weights, biases, gradients_w, gradients_b):
        for i in range(len(weights)):
            weights[i] -= self.lr * (gradients_w[i] + self.weight_decay * weights[i])
            biases[i] -= self.lr * gradients_b[i]  # Biases are not regularized

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v_w = None
        self.v_b = None
    
    def update(self, weights, biases, gradients_w, gradients_b):
        if self.v_w is None:
            self.v_w = [np.zeros_like(w) for w in weights]
            self.v_b = [np.zeros_like(b) for b in biases]
        
        for i in range(len(weights)):
            self.v_w[i] = self.momentum * self.v_w[i] - self.lr * (gradients_w[i] + self.weight_decay * weights[i])
            self.v_b[i] = self.momentum * self.v_b[i] - self.lr * gradients_b[i]
            weights[i] += self.v_w[i]
            biases[i] += self.v_b[i]

class Nesterov(Optimizer):
    def __init__(self, learning_rate=1e-3, momentum=0.9, weight_decay=0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v_w = None
        self.v_b = None
    
    def update(self, weights, biases, gradients_w, gradients_b):
        if self.v_w is None:
            self.v_w = [np.zeros_like(w) for w in weights]
            self.v_b = [np.zeros_like(b) for b in biases]
        
        for i in range(len(weights)):
            lookahead_w = weights[i] - self.momentum * self.v_w[i]
            lookahead_b = biases[i] - self.momentum * self.v_b[i]
            
            self.v_w[i] = self.momentum * self.v_w[i] - self.lr * (gradients_w[i] + self.weight_decay * lookahead_w)
            self.v_b[i] = self.momentum * self.v_b[i] - self.lr * gradients_b[i]
            
            weights[i] = lookahead_w + self.v_w[i]
            biases[i] = lookahead_b + self.v_b[i]

class AdaGrad(Optimizer):
    def __init__(self, learning_rate=1e-2, epsilon=1e-7, weight_decay=0.0):
        self.lr = learning_rate
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.v_w = None
        self.v_b = None
    
    def update(self, weights, biases, gradients_w, gradients_b):
        if self.v_w is None:
            self.v_w = [np.zeros_like(w) for w in weights]
            self.v_b = [np.zeros_like(b) for b in biases]
        
        for i in range(len(weights)):
            self.v_w[i] += gradients_w[i] ** 2
            self.v_b[i] += gradients_b[i] ** 2
            weights[i] -= self.lr * (gradients_w[i] + self.weight_decay * weights[i]) / (np.sqrt(self.v_w[i]) + self.epsilon)
            biases[i] -= self.lr * gradients_b[i] / (np.sqrt(self.v_b[i]) + self.epsilon)

class RMSProp(Optimizer):
    def __init__(self, learning_rate=1e-3, beta=0.9, epsilon=1e-7, weight_decay=0.0):
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.v_w = None
        self.v_b = None
    
    def update(self, weights, biases, gradients_w, gradients_b):
        if self.v_w is None:
            self.v_w = [np.zeros_like(w) for w in weights]
            self.v_b = [np.zeros_like(b) for b in biases]
        
        for i in range(len(weights)):
            self.v_w[i] = self.beta * self.v_w[i] + (1 - self.beta) * (gradients_w[i] ** 2)
            self.v_b[i] = self.beta * self.v_b[i] + (1 - self.beta) * (gradients_b[i] ** 2)
            weights[i] -= self.lr * (gradients_w[i] + self.weight_decay * weights[i]) / (np.sqrt(self.v_w[i]) + self.epsilon)
            biases[i] -= self.lr * gradients_b[i] / (np.sqrt(self.v_b[i]) + self.epsilon)

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m_w, self.v_w, self.m_b, self.v_b = None, None, None, None
        self.t = 0
   
    def update(self, weights, biases, gradients_w, gradients_b):
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in weights]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_b = [np.zeros_like(b) for b in biases]
        
        self.t += 1
        
        for i in range(len(weights)):
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gradients_w[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (gradients_w[i] ** 2)
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gradients_b[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (gradients_b[i] ** 2)
            
            weights[i] -= self.lr * (self.m_w[i] / (np.sqrt(self.v_w[i]) + self.epsilon) + self.weight_decay * weights[i])
            biases[i] -= self.lr * (self.m_b[i] / (np.sqrt(self.v_b[i]) + self.epsilon))


class Nadam(Optimizer):
    def __init__(self, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m_w, self.v_w, self.m_b, self.v_b = None, None, None, None
        self.t = 0

    def update(self, weights, biases, gradients_w, gradients_b):
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in weights]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_b = [np.zeros_like(b) for b in biases]
        self.t += 1

        for i in range(len(weights)):
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gradients_w[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (gradients_w[i] ** 2)
            m_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            weights[i] -= self.lr * ((self.beta1 * m_hat + (1 - self.beta1) * gradients_w[i]) / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * weights[i])

            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gradients_b[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (gradients_b[i] ** 2)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)
            biases[i] -= self.lr * ((self.beta1 * m_hat_b + (1 - self.beta1) * gradients_b[i]) / (np.sqrt(v_hat_b) + self.epsilon))

optimizers = {
    "sgd": SGD,
    "momentum": Momentum,
    "nestrov": Nesterov,
    "adagrad": AdaGrad,
    "rmsprop": RMSProp,
    "adam": Adam,
    "nadam": Nadam,
}
