import numpy as np
from models.neural_networks import NeuralNetwork


def main():
    model = NeuralNetwork(7)
    model.add(NeuralNetwork.layer(5, 3), activation="relu")
    model.add(NeuralNetwork.layer(3, 5), activation="relu")
    model.add(NeuralNetwork.layer(1, 3), activation="sigmoid")


if __name__ == "__main__":
    main()
