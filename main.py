<<<<<<< HEAD
import numpy 
from models.neural_networks import NeuralNetwork

=======
import numpy as np
from models.neural_networks import NeuralNetwork


>>>>>>> 3fecd1c8eb0af2f424b867a2d32fa4ae933bd5eb
def main():
    model = NeuralNetwork(7)
    model.add(NeuralNetwork.layer(5, 3), activation="relu")
    model.add(NeuralNetwork.layer(3, 5), activation="relu")
    model.add(NeuralNetwork.layer(1, 3), activation="sigmoid")
<<<<<<< HEAD
    model.train(X, y, epochs=20, learning_rate=0.3)
=======

>>>>>>> 3fecd1c8eb0af2f424b867a2d32fa4ae933bd5eb

if __name__ == "__main__":
    main()
