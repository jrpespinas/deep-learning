"""Neural Networks in Numpy"""

import numpy as np

class NeuralNetwork:
    def __init__(self):
        pass
    
    def initialize_parameters(self):
        pass

    def update_parameters(self):
        pass

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

def main():
    pass

if __name__ == "__main__":
    main()