import numpy as np

class sigmoid():
    """
    An object that allows for easy access to all activation functions needed for
    sigmoid activation.

    Methods:
        forward : gives the result of a sigmoid avtivation
        derviative : gives the result for backpropagation with sigmoid activation
    """

    def forward(self, inputs):
        return 1 / (1 + np.exp(-inputs))
    
    def derivative(self, inputs):
        return np.exp(-inputs) / np.square(1 + np.exp(-inputs))

if __name__ == "__main__":
    """
    If this file is being run individually test the implementation of neural net classes.
    """

    test_inputs = np.array([[2], [1]])
    test_activation = sigmoid()
    print("Sigmoid Foward Results:", test_activation.forward(test_inputs))
    print("Sigmoid Backwards Results:", test_activation.derivative(test_inputs))
    