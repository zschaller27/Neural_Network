import numpy as np

from nets import *
from loss_functions import *

def perturb_train(net, train_data, ground_truth_y, loss_function, epochs=1, step_size=0.0001):
    # For each epoch perturbe the each weight in the network and compare the loss values
    for epoch in range(epochs):
        # Find initial loss value
        y_hat = net.forward(train_data)
        initial_loss = loss_function.error(y_hat, ground_truth_y)

        # Go through each hidden layer
        for i_layer in range(net.n):
            # Go through each neuron in the layer
            for i_neuron in range(net.hidden_layers[i_layer].m):
                # Once again iterate through the weights changing them by the step_size
                # both positive and negative. Choosing the value which creates the smaller
                # loss.
                for i_weight in range(len(net.hidden_layers[i_layer].neurons[i_neuron].weights)):
                    print(net.hidden_layers[i_layer].neurons[i_neuron].weights[i_weight])

if __name__ == "__main__":
    """
    If this file is being run individually test the implementation of training functions.
    """

    test_inputs = np.array([[2], [1]])

    ## Test perturb_train Function Implementation ##
    print("Testing perturb_train")
    test_network = LinearNeuralNetwork(2, 1, 2, 1, sigmoid())
    perturb_train(test_network, test_inputs, [1], MeanSquaredError())
