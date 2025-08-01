import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    stable = x - np.max(x, axis=0, keepdims=True)
    exponentials = np.exp(stable)
    return exponentials / np.sum(exponentials, axis=0, keepdims=True)

def accuracy(output, label):
    pred_class = np.argmax(output)
    true_class = np.argmax(label)
    return 1 if pred_class == true_class else 0
