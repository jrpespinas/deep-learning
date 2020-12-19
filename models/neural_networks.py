"""Neural Networks in Numpy"""

import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.architecture = []

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

        return [weights, bias]


def main():
    model = NeuralNetwork()
    model.add(NeuralNetwork.layer(15, 15))


if __name__ == "__main__":
    main()
