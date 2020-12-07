"""Shallow Neural Network in Numpy"""

import numpy as np
from typing import Tuple


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


def layer_size(X, Y, hidden_layer: int = 5) -> Tuple[int, int, int]:
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


def initialize_parameters(input_layer, hidden_layer, output_layer,
                          seed: int = 69, constant: float = 0.01) -> dict:
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


def forward_propagation(X, parameters, activation) -> Tuple[float, dict]:
    """
    Make predictions

    Parameters
    ----------
    X : numpy.ndarray
        Data
    parameters : dict
        Weights and bias
    activation : function
        Activation function

    Returns
    -------
    A2 : numpy.ndarray
        Predictions
    cache : dict
        cache of the computations
    """
    weights_1 = parameters["weights_1"]
    bias_1 = parameters["bias_1"]
    weights_2 = parameters["weights_2"]
    bias_2 = parameters["bias_2"]

    Z1 = np.dot(weights_1, X) + bias_1
    A1 = activation(Z1)
    Z2 = np.dot(weights_2, A1) + bias_2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache


def cost_function(predictions, labels, parameters) -> float:
    """
    Compute the cost function

    Parameters
    ----------
    predictions : numpy.ndarray
        Predicted output from `forward_propagation`
    labels : numpy.ndarray
        Ground truth of the predictions
    parameters : dict
        Weights

    Returns
    -------
    cost : float
        Loss or cost 
    """
    num_training_examples = labels.shape[1]

    log_1 = np.multiply(labels, np.log(predictions))
    log_0 = np.multiply((1 - labels), np.log(1 - predictions))
    cost = (-1 / num_training_examples) * np.sum(log_1 + log_0)

    cost = np.squeeze(cost)

    return cost


def backward_propagation(X, labels, parameters, cache) -> dict:
    num_training_examples = X.shape[1]

    weights_1 = parameters["weights_1"]
    weights_2 = parameters["weights_2"]
    A1 = cache['A1']
    A2 = cache['A2']

    d_output_layer = A2 - labels
    d_weights_2 = (1 / num_training_examples) * np.dot(d_output_layer, A1.T)
    d_bias_2 = (1 / num_training_examples) * \
        np.sum(d_output_layer, axis=1, keepdims=True)
    d_hidden_layer = np.dot(weights_2.T, d_output_layer) * relu_prime(A1)
    d_weights_1 = (1 / num_training_examples) * np.dot(d_hidden_layer, X.T)
    d_bias_1 = (1 / num_training_examples) * \
        np.sum(d_hidden_layer, axis=1, keepdims=True)

    grads = {
        "d_weights_1": d_weights_1,
        "d_bias_1": d_bias_1,
        "d_weights_2": d_weights_2,
        "d_bias_2": d_bias_2
    }

    return grads


def main():
    pass


if __name__ == "__main__":
    main()
