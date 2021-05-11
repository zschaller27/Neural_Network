import numpy as np

from layers import *

class LinearNeuralNetwork:
    """
    A object that combines a neruon objects into hidden layers to create a neural network.
    
    Attributes:
        n : the number of expected values in the input to the network.
        hidden_layers : A set of 'l' layers made up of 'm' neurons.
    """

    def __init__(self, num_inputs, l, m, o):
        """
        Initialize the network with 'l' hidden layers (including the input layer) each layer having 'm'
        neurons. The output layer has 'o' neurons. By default this uses MSE loss.

        Parameters:
            num_inputs : is the number of values that are in the input vector
            l : the number of hidden layers to make.
            m : the number of neurons per layer.
            o : the number of neurons on the output layer.
        """
        # Verify that the numbers given are valid
        assert l > 0
        assert m > 0
        assert o > 0

        # Initialize hidden_layers list to an empty list
        self.hidden_layers = []
        
        # Add the input layer
        self.n = num_inputs
        self.hidden_layers.append(LinearLayer(self.n, 1, m))
        
        # Add the rest of the hidden layers
        for _ in range(l - 1):
            self.hidden_layers.append(LinearLayer(m, 1, m))
        
        # Add the output layer
        self.hidden_layers.append(LinearLayer(m, 1, o))

    def forward(self, input):
        """
        Provides an output numpy matrix from the layer for a give set of inputs.

        Paramters:
            inputs : the numpy matrix of inputs with shape n x 1, where n is the designated num_inputs
                     from the __init__ function.
        
        Return:
            A numpy matrix with shape o x 1, where o is the number of neurons on the layer,
            with each index being the output from that node
        """
        # Verify that that the input has the right shape
        assert input.shape == (self.n, 1)

        x = input
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i].forward(x)
        return x

# TODO: Write a modular train function that can train any network structure with a given test data set

if __name__ == "__main__":
    """
    If this file is being run individually test the implementation of neural net classes.
    """

    test_inputs = np.array([[2], [1]])

    ## Test NeuralNet Class Implementation ##
    print("Testing Neural Net")
    test_network = LinearNeuralNetwork(2, 1, 2, 1)
    print(test_network.forward(test_inputs))