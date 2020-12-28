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
    """
    def __init__(self, seed, architecture):
        super().__init__()
        np.random.seed(seed)
        self.architecture = architecture
    
    def initialize_layers(self, constant: float = 0.01):
        number_of_layers = len(self.architecture)
        parameters = {}

        for layer_num, layer in enumerate(self.architecture):
            layer_num += 1
            input_dims = layer["input_dim"]
            output_dims = layer["output_dim"]

            parameters["W" + str(layer_num)] = np.random.randn(output_dims,
                input_dims) * constant
            parameters["b" + str(layer_num)] = np.zeros((output_dims, 1))

        return parameters

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
            Z = self.cache["Z" + str(layer_num)]
            A_previous = self.cache["A" + str(layer_num-1)]
            activation = layer["activation" + str(layer_num)]
            weights = layer["W" + str(layer_num)]
            bias = layer["b" + str(layer_num)]
            
            dA_previous, dW, db = self._backward_step(A_previous, dA_current, 
                m, weights, Z, activation)
            
            updated_weights, updated_bias = self._update_parameters(dW, db, 
                weights, bias, learning_rate)

            self.architecture[layer_num - 1]["W" + str(layer_num)] = updated_weights
            self.architecture[layer_num - 1]["b" + str(layer_num)] = updated_bias
        

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
<<<<<<< HEAD
    print("Hello, this is the Neural Network Class script")
    
=======
   print("Hello, this is the Neural Network class script.") 
>>>>>>> 3fecd1c8eb0af2f424b867a2d32fa4ae933bd5eb

if __name__ == "__main__":
    main()
