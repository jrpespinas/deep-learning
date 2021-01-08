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
        """Initializes layers by adding random values to the layers
        given by `self.architecture`.

        Parameters:
            constant (:obj:`float`, optional): Value for scaling 
                random numbers.

        Returns:
            parameters (dict): The randomly initialized weights and biases.
        """
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

    def forward_propagation(self, X, parameters):
        """
        Predicts the values for X.

        Args:
            X (numpy.ndarray): The input features or dataset.
            parameters (dict): The weights and biases.

        Returns:
            A_current (numpy.ndarray): The predicted value.
            cache (dict): Stored computations for backpropagation.
        """
        cache = {}
        A_current = X

        for layer_num, layer in enumerate(self.architecture):
            layer_num += 1

            A_previous = A_current
            weights = parameters["W" + str(layer_num)]
            bias = parameters["b" + str(layer_num)]
            activation = layer["activation"]

            assert activation in self.activation_functions, \
                "ERROR: Activation not supported"
                
            A_current, Z_current = self._forward_step(weights, A_previous,
                                                      bias, activation)

            cache["A" + str(layer_num)] = A_previous
            cache["Z" + str(layer_num)] = Z_current

        return A_current, cache

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

    def backward_propagation(self, y, y_hat, cache, parameters):
        """Computes the gradients with respect to the loss value.

        Args:
            y (numpy.ndarray): The true labels or ground truth.
            y_hat (numpy.ndarray): Predictions of the model.
            cache (dict): The stored computations from forward propagation.
            parameters (dict): The weights and biases

        Returns: 
            gradients (dict): The derivatives computed.
        """
        gradients = {}
        m = y.shape[1]
        y = y.reshape(y_hat.shape)

        dA_previous = - (np.divide(y, y_hat) + np.divide(1 - y, 1 - y_hat))

        for layer_num, layer in reversed(list(enumerate(self.architecture))):
            layer_num += 1

            activation = layer["activation"]
            dA_current = dA_previous

            A_previous = cache["A" + str(layer_num - 1)]
            Z_current = cache["Z" + str(layer_num)]
            weights = parameters["W" + str(layer_num)]
            
            dA_previous, dW, db = self._backward_step(A_previous, dA_current,
                                                      weights, Z_current, 
                                                      activation)
            
            gradients["dW" + str(layer_num)] = dW
            gradients["db" + str(layer_num)] = db
        
        return gradients

    def train(self, X, y, epochs: int, learning_rate: float = 0.1, 
              verbosity: bool = True):
        """Gradient Descent

        Args:
            X (numpy.ndarray): The input features or dataset.
            y (numpy.ndarray): The true labels or ground truth.
            epochs (int): number of training iterations 
                or number of steps of gradient descent.
            learning_rate (:obj:`float`, optional): step size 
                for updating the weight and bias.
            verbosity (True): Displays loss during training.

        Returns:
            parameters (dict): The weights and biases
            loss_history (list): Recorded loss per epoch
        """
        parameters = initialize_layers()
        loss_history = []
        accuracy_history = []

        for i in range(epochs):
            y_hat, cache = self.forward_propagation(X, parameters)
            loss = self.cost_function(y, y_hat)
            loss_history.append(loss)

            gradients = self.backward_propagation(y, y_hat, cache, parameters)
            parameters = self.update_parameters(parameters, gradients, learning_rate)

        return parameters, loss_history 

    def _forward_step(self, W, A, b, activation):
        if activation == "relu":
            g = self.relu
        elif activation == "sigmoid":
            g = self.sigmoid

        Z = np.dot(W, A) + b
        A = g(Z)

        return A, Z

    def _backward_step(self, A_previous, dA, W, Z, activation):
        m = A_previous.shape[1]

        if activation == "relu":
            g_prime = self.relu_prime
        elif activation == "sigmoid":
            g_prime = self.sigmoid_prime
        
        dZ = np.multiply(dA, g_prime(Z))
        dW = np.dot(dZ, A_previous.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m 
        dA_previous = np.dot(W.T, dZ)

        return dA_previous, dW, db

    def update_parameters(self, parameters, gradients, 
        learning_rate: float = 0.1):
        """Updates the parameters after a successful backpropagation.

        Parameters:
            parameters (dict): The weights and biases.
            gradients (dict): The computed derivatives.
            learning_rate (:obj:`float`, optional): step size 
                for updating the weight and bias.

        Returns:
            parameters (dict): The weights and biases.
        """
        for layer_num, layer in enumerate(self.architecture):
            parameters["W" + str(layer_num)] -= learning_rate * gradients["dW" + str(layer_num)]
            parameters["b" + str(layer_num)] -= learning_rate * gradients["db" + str(layer_num)]

        return parameters
        

def main():
    print("Hello, this is the Neural Network Class script")
    

if __name__ == "__main__":
    main()
