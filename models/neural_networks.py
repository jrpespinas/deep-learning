"""Neural Networks in Numpy"""

import numpy as np


class NeuralNetwork:
    def __init__(self, seed):
        np.random.seed(seed)
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
        """
        Initialize a hidden layer with the size (`output_dims`, `input_dims`)

        Parameters
        ----------
        output_dims : int
            Number of hidden units at the current layer
        input_dims : int
            Number of hidden units at the previous layer
        constant : float
            Number to scale down weights

        Returns
        -------
        parameters : list
            List containing the intialized weights and bias
        """
        weights = np.random.randn(output_dims, input_dims) * constant
        bias = np.zeros((output_dims, 1))

        parameters = [weights, bias]

        return parameters

    def add(self, parameters, activation="relu"):
        """
        Add the weights `layer()` to the architecture `self.architecture`

        Parameters
        ----------
        parameters : numpy.ndarray
            Weights and bias
        activation : str
            Activation function 

        """
        assert (activation == "relu") or (activation == 'sigmoid'), \
            "ERROR: Activation not supported"

        current_weights = parameters[0]
        current_bias = parameters[1]

        layer_num = len(self.architecture) + 1
        self.architecture.append({
            "W" + str(layer_num): current_weights,
            "b" + str(layer_num): current_bias,
            "activation" + str(layer_num): activation
        })

        if layer_num >= 2:
            previous_weights = self.architecture[-2]["W" + str(layer_num - 1)]
            self._check_dimensions(
                current_weights, previous_weights, layer_num)

    def forward_propagation(self, X):
        A_current = X

        for layer_num, layer in enumerate(self.architecture):
            layer_num += 1

            A_previous = A_current
            weights = layer["W" + str(layer_num)]
            bias = layer["b" + str(layer_num)]
            activation = layer["activation"]

            A_current, Z_current = self._forward_step(weights, A_previous,
                                                      bias, activation)

            self.cache.append({
                "A" + str(layer_num): A_current,
                "Z" + str(layer_num): Z_current
            })

        return A_current

    def _check_dimensions(self, current_layer, previous_layer, layer_num):
        assert current_layer.shape[1] == previous_layer.shape[0], \
            "ERROR: Dimensions at layer %d and layer %d are not compatible!" % (
                layer_num, layer_num-1
        )

    def _forward_step(self, W, A, b, activation):
        Z = np.dot(W, A) + b

        if activation == "relu":
            g = self.relu
        elif activation == "sigmoid":
            g = self.sigmoid

        A = g(Z)

        return A, Z


def main():
    model = NeuralNetwork(7)
    model.add(NeuralNetwork.layer(5, 21), activation="relu")
    model.add(NeuralNetwork.layer(3, 5), activation="relu")
    model.add(NeuralNetwork.layer(3, 3), activation="relu")
    model.forward_propagation()


if __name__ == "__main__":
    main()
