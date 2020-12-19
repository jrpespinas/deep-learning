"""Neural Networks in Numpy"""

import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.architecture = []
        self.cache = []

    def relu(self, z):
        return np.maximum(0, z)

    def sigmoid(self, z):
        return 1 / np.add(1, np.exp(-z))

    def relu_prime(self, z):
        z[z <= 0] = 0
        z[z > 0] = 1
        return z

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    @staticmethod
    def layer(output_dims, input_dims, constant: float = 0.01):
        weights = np.random.randn(output_dims, input_dims) * constant
        bias = np.zeros((output_dims, 1))

        parameters = [weights, bias]

        return parameters

    def add(self, parameters, activation="relu"):
        assert (activation == "relu") or (activation == 'sigmoid'), \
            "ERROR: Activation not supported"

        layer_num = len(self.architecture) + 1
        self.architecture.append({
            "W" + str(layer_num): parameters[0],
            "b" + str(layer_num): parameters[1],
            "activation" + str(layer_num): activation
        })

    def __forward_step(self, W, A, b, activation):
        Z = np.dot(W, A) + b

        if activation == "relu":
            g = self.relu
        elif activation == "sigmoid":
            g = self.sigmoid

        A = g(Z)

        return A, Z


def main():
    model = NeuralNetwork()
    model.add(NeuralNetwork.layer(5, 21), activation="relu")


if __name__ == "__main__":
    main()
