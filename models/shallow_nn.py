"""Shallow Neural Network in Numpy"""

import numpy as np


def tanh(z):
    numerator = np.subtract(np.exp(x), np.exp(-x))
    denominator = np.add(np.exp(x), np.exp(-x))
    return numerator / denominator


def sigmoid(z):
    numerator = 1
    denominator = np.add(1, np.exp(-z))
    return numerator / denominator


def main():
    pass


if __name__ == "__main__":
    main()
