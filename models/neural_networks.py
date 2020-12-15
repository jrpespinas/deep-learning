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
        
def main():
    pass

if __name__ == "__main__":
    main()