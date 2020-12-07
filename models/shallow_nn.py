"""Shallow Neural Network in Numpy"""

import numpy as np


def relu_prime(z):
    return 1 if z >= 0 else 0


def tanh_prime(z):
    return 1 - np.power(tanh(z), 2)


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
    """
    Get the layer sizes

    Parameters
    ----------
    X : numpy.ndarray
        Features
    Y : numpy.ndarray
        Labels
    hidden_layer : int
        Number of Neurons

    Returns
    -------
    input_layer : int 
        Size of input layer
    hidden_layer : int 
        Number of Neurons within one hidden layer
    output_layer : int
        Size of output layer
    """
    input_layer = X.shape[0]
    hidden_layer = hidden_layer
    output_layer = Y.shape[0]
    return input_layer, hidden_layer, output_layer


def initialize_parameters(input_layer, hidden_layer, output_layer, seed: int = 69, constant: float = 0.01):
    """
    Initialize the parameters of all the layers

    Parameters
    ----------
    input_layer : int 
        Size of input layer
    hidden_layer : int 
        Number of Neurons within one hidden layer
    output_layer : int
        Size of output layer
    seed : int
        Seed number
    constant : float
        Constant to normalize weights

    Returns
    -------
    parameters : dict
        Dictionary of the initialized parameters
    """
    np.random.seed(seed)

    weights_1 = np.random.randn(input_layer, hidden_layer) * constant
    bias_1 = np.zeros((hidden_layer, 1))
    weights_2 = np.random.randn(output_layer, hidden_layer) * constant
    bias_2 = np.zeros((output_layer, 1))

    parameters = {
        "weights_1": weights_1,
        "bias_1": bias_1,
        "weights_2": weights_2,
        "bias_2": bias_2
    }

    return parameters


def main():
    pass


if __name__ == "__main__":
    main()
