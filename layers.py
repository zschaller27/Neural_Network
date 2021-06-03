import numpy as np

from neurons import *

class LinearLayer:
    """
    Combines multiple neuron objects into a linear hidden layer with a given number of neurons.
    
    Attributes:
        neurons : A list of all neurons in the layer
    
    Methods:
        forward : takes in a number of input values to pass thorugh the layer and generate an output
                  with shape m x 1 where m is the number of neurons in the layer.
    """
    def __init__(self, num_inputs, bias, m):
        """
        Intialize the layer's neurons and save them to a list.

        Paramters:
            num_inputs : the number of inputs to each neuron in the layer
            bias : the bias value for the neurons in the layer
            m : the number of neurons in the layer
        """
        self.m = m
        self.neurons = []
        for _ in range(self.m):
            self.neurons.append(Neuron(num_inputs, bias))

    def forward(self, inputs):
        """
        Provides an output numpy matrix from the layer for a give set of inputs.

        Paramters:
            inputs : the numpy matrix of inputs with shape (n x 1)
        
        Return:
            A numpy matrix with shape m x 1, where m is the number of neurons on the layer,
            with each index being the output from that node
        """
        ret = np.zeros((self.m, 1))

        for i in range(self.m):
            ret[i] = self.neurons[i].forward(inputs)

        return ret

if __name__ == "__main__":
    """
    If this file is being run individually test the implementation of layer classes.
    """

    test_inputs = np.array([[2], [1]])

    ## Test LinearLayer Class Implementation ##
    print("Testing Single Layer")
    test_layer = LinearLayer(2, 1, 3)
    print(test_layer.forward(test_inputs))
