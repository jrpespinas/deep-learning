"""Shallow Neural Network in Numpy"""

import numpy as np


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


def main():
    pass


if __name__ == "__main__":
    main()
