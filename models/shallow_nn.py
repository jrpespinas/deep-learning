"""Shallow Neural Network in Numpy"""

import numpy as np


def relu_prime(z):
    return 1 if z >= 0 else 0


def tanh_prime(z):
    return 1 - np.square(tanh(z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return np.maximum(0, z)


def tanh(z):
    numerator = np.subtract(np.exp(z), np.exp(-z))
    denominator = np.add(np.exp(z), np.exp(-z))
    return numerator / denominator


def sigmoid(z):
    numerator = 1
    denominator = np.add(1, np.exp(-z))
    return numerator / denominator


def layer_size(X, Y, hidden_layer: int = 5):
    input_layer = X.shape[0]
    hidden_layer = hidden_layer
    output_layer = Y.shape[0]
    return (input_layer, hidden_layer, output_layer)


def main():
    pass


if __name__ == "__main__":
    main()
