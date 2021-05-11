import numpy as np

class Neuron:
    """
    A neuron object holds all information needed to compute the ouput from a single neuron.

    Attributes:
        weights : numpy matrix to hold the weight values for the neuron with shape (n x 1).
        bias : the bias term used in output computation.
    
    Methods:
        forward : computes the neuron output for a given numpy matrix of inputs (with shape n x 1)
    """

    def __init__(self, num_inputs, b=1):
        """
        The initialization function for the neuron. Will generate a weight matrix using near 0 values.

        Paramters:
            num_inputs : the number of input values into the neuron.
            b : The given bias value to use for this neuron.
        """
        self.weights = np.zeros_like(None, shape=(num_inputs, 1))
        self.bias = b
        
    def forward(self, inputs):
        """
        Computs the output from this neruon for the given inputs.

        Paramters:
            inputs : the numpy matrix of inputs with shape (n x 1)
        
        Return:
            A single value in a numpy array which is the output from the neuron object
        """

        # Verify that the inputs have the correct shape
        assert inputs.shape == self.weights.shape

        return np.reshape(np.dot(np.transpose(inputs), self.weights), 1) + self.bias

if __name__ == "__main__":
    """
    If this file is being run individually test the implementation of neuron classes.
    """

    test_inputs = np.array([[2], [1]])

    ## Test Linear Neuron Class Implementation ##
    print("Testing Single Linear Neuron")
    test_neuron = Neuron(2, 1)
    print(test_neuron.forward(test_inputs))
