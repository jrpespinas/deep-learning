"""Neural Networks in Numpy"""

import numpy as np


class NeuralNetwork:
    def __init__(self):
        pass

    def __relu(self, z):
        return np.maximum(0, z)

    def __sigmoid(self, z):
        return 1 / np.add(1, np.exp(-z))

    def __relu_prime(self, z):
        z[z <= 0] = 0
        z[z > 0] = 1
        return z

    def __sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))


def main():
    pass


if __name__ == "__main__":
    main()
