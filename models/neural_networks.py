"""Neural Network in Numpy"""

import numpy as np


class Activation:
    def __init__(self):
        self.activation_functions = [
            "sigmoid",
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
        super().__init__()
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
        assert activation in self.activation_functions, \
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

    def backward_propagation(self, y, y_hat, learning_rate: float = 0.01):
        """Computes the gradients with respect to the loss value.

        Args:
            y (numpy.ndarray): The true labels or ground truth.
            y_hat (numpy.ndarray): Predictions of the model.
            learning_rate (:obj:`float`, optional): step size
                for updating the weight and bias.
        """
        m = y.shape[1]

        dA_previous = - (np.divide(y, y_hat) + np.divide(1 - y, 1 - y_hat))
        for layer_num, layer in reversed(list(enumerate(self.architecture))):
            layer_num += 1

            dA_current = dA_previous
            Z = self.cache[layer_num - 1]["Z" + str(layer_num)]
            A_previous = self.cache[layer_num - 1]["A" + str(layer_num-1)]
            activation = layer["activation" + str(layer_num)]
            weights = layer["W" + str(layer_num)]
            bias = layer["b" + str(layer_num)]

            dA_previous, dW, db = self._backward_step(A_previous, dA_current,
                                                      m, weights, Z, activation)

            new_weights, new_bias = self._update_parameters(dW, db, weights,
                                                            bias, learning_rate)

            self.architecture[layer_num - 1]["W" +
                                             str(layer_num)] = new_weights
            self.architecture[layer_num - 1]["b" +
                                             str(layer_num)] = new_bias

    def train(self, X, y, epochs: int, learning_rate: float = 0.01,
              verbosity: bool = True):
        """Gradient Descent

        Args:
            X (numpy.ndarray): The input features or dataset.
            y (numpy.ndarray): The true labels or ground truth.
            epochs (int): number of training iterations 
                or number of steps of gradient descent.
            learning_rate (float): step size for updating the weight and bias.
            verbosity (True): Displays loss during training.
        """
        loss_history = []

        for i in range(epochs):
            predictions = self.forward_propagation(X)
            loss = self.cost_function(y, predictions)
            loss_history.append(loss)
            self.backward_propagation(y, predictions, learning_rate)

            if verbosity:
                print("Loss: {:.5f}".format(loss))

    def _check_dimensions(self, current_layer, previous_layer, layer_num):
        assert current_layer.shape[1] == previous_layer.shape[0], \
            "ERROR: Dimensions at layer %d and layer %d are not compatible!" % (
                layer_num, layer_num-1
        )

    def _forward_step(self, W, A, b, activation):
        if activation == "relu":
            g = self.relu
        elif activation == "sigmoid":
            g = self.sigmoid

        Z = np.dot(W, A) + b
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

    def _update_parameters(self, dW, db, W, b, learning_rate):
        W -= learning_rate * dW
        b -= learning_rate * db
        return W, b


def main():
    print("Hello, this is the Neural Network class script.")


if __name__ == "__main__":
    main()
