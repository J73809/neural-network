import numpy as np
from src.activations import relu, relu_derivative, softmax

class NeuralNetwork():
    def __init__(self, layer_sizes):
        self.size = layer_sizes
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i+1]
            weight = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
            bias = np.zeros((output_size, 1))
            
            self.weights.append(weight)
            self.biases.append(bias)
            
    def forward(self, x):
        activations = [x]
        zs = []
        
        a = x
        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], a) + self.biases[i]
            zs.append(z)
            a = relu(z)
            activations.append(a)

        z = np.dot(self.weights[-1], a) + self.biases[-1]
        zs.append(z)
        output = softmax(z)
        activations.append(output)
        
        return output, activations, zs
    
    def backward(self, x, y, activations, zs, learning_rate):
        L = len(self.weights)
    
        dW = [np.zeros_like(w) for w in self.weights]
        dB = [np.zeros_like(b) for b in self.biases]

        delta = activations[-1] - y 
    
        dW[-1] = np.dot(delta, activations[-2].T)
        dB[-1] = delta
    
        for l in range(L - 2, -1, -1):
            delta = np.dot(self.weights[l+1].T, delta) * relu_derivative(zs[l])
            dW[l] = np.dot(delta, activations[l].T)
            dB[l] = delta

        for i in range(L):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * dB[i]
