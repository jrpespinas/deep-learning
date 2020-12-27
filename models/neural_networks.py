"""Neural Network in Numpy"""

import numpy as np

class Activation:
    def __init__(self):
        self.activation_funcitons = [
            "sigmod",
            "relu"
        ]

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

class NeuralNetwork(Activation):
    """Neural Network class implemented in Numpy.

    The Neural Network class implemented in Numpy 
    is loosely inspired by TensorFlow and PyTorch 
    implementation.

    Note:
        More methods and functionality will be included in future works.
    
    Args:
        seed (int): initialize pseudo-random number generation.

    Attributes:
        architecture (:obj:`list` of :obj:`dict`): The structure 
            of the Neural Network.
        cache (:obj:`list` of :obj:`float`): Temporary storage of computations
            for backward propagation.
    """
    def __init__(self, seed):
        np.random.seed(seed)
        self.architecture = []
        self.cache = []

    @staticmethod
    def layer(output_dims, input_dims, constant: float = 0.01):
        """Initializes random weights of a single hidden layer.

        Args:
            output_dims (int): Number of hidden units at the current layer.
            input_dims (int): Number of hidden units at the previous layer.
            constant (:obj:`float`, optional): Number to scale down weights.

        Returns:
            parameters (:obj:`list` of :obj:`numpy.ndarray`): List containing 
                the intialized weights and bias
        """
        weights = np.random.randn(output_dims, input_dims) * constant
        bias = np.zeros((output_dims, 1))

        parameters = [weights, bias]

        return parameters

    def add(self, parameters, activation="relu"):
        """
        Adds the weights the initialized layers to the architecture.

        Note: 
            Change the activation function at the output layer.

        Args:
            parameters (numpy.ndarray): Weights and bias.
            activation (:obj:`str`, optional): Activation function.

        Example:
            >>> model = NeuralNetwork()
            >>> model.add(NeuralNetwork.layer(5, 21), activation="relu")
            >>> model.add(NeuralNetwork.layer(1, 5), activation="sigmoid")
        """
        assert activation in self.activation_funcitons, \
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
        """
        Predicts the values for X.

        Args:
            X (numpy.ndarray): The input features or dataset.

        Returns:
            A_current (numpy.ndarray): The predicted value.
        """
        A_current = X

        for layer_num, layer in enumerate(self.architecture):
            layer_num += 1

            A_previous = A_current
            weights = layer["W" + str(layer_num)]
            bias = layer["b" + str(layer_num)]
            activation = layer["activation" + str(layer_num)]
            print(weights)

            A_current, Z_current = self._forward_step(weights, A_previous,
                                                      bias, activation)

            self.cache.append({
                "A" + str(layer_num): A_current,
                "Z" + str(layer_num): Z_current
            })

        return A_current

    def cost_function(self, y, y_hat) -> float:
        """
        Computes the loss of the model.

        Args:
            y (numpy.ndarray): The true labels or ground truth.
            y_hat (numpy.ndarray): Predictions of the model.

        Retruns:
            total_loss (float): The loss of the model.
        """
        m = y.shape[1]

        loss = np.multiply(y, np.log(y_hat)) + \
            np.multiply((1 - y), np.log(1 - y_hat))
        total_loss = np.sum(loss) / m

        total_loss = np.squeeze(total_loss)

        return total_loss

    def backward_propagation(self, dA):
        raise NotImplementedError

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

    def _backward_step(self, A_previous, dA, m, W, Z, activation):
        if activation == "relu":
            g_prime = self.relu_prime
        elif activation == "sigmoid":
            g_prime = self.sigmoid_prime
        
        dZ = np.multiply(dA, g_prime(Z))
        dW = np.dot(dZ, A_previous.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_previous = np.dot(W.T, dZ)

        return dA_previous, dW, db

    def _update_parameters(self, dW, db):
        raise NotImplementedError


def main():
    X = np.random.randn(3, 1)
    model = NeuralNetwork(7)
    model.add(NeuralNetwork.layer(5, 3), activation="relu")
    model.add(NeuralNetwork.layer(3, 5), activation="relu")
    model.add(NeuralNetwork.layer(1, 3), activation="sigmoid")
    model.forward_propagation(X)


if __name__ == "__main__":
    main()
