import numpy as np

from nets import *
from loss_functions import *

def perturb_train(net, train_data, ground_truth_y, loss_function, epochs=1, step_size=0.0001):
    # Initialize a value to use for convergence checking
    prev_loss = float("inf")

    # For each epoch perturbe the each weight in the network and compare the loss values
    for epoch in range(epochs):
        ## Test Code ##
        print("---------- Running Epoch %d ----------" %epoch)
        printWeights(net)
        ## End Test Code ##

        # Find initial loss value
        y_hat = net.forward(train_data)
        initial_loss = loss_function.error(y_hat, ground_truth_y)

        ## Test Code ##
        for i in range(len(ground_truth_y)):
            print("%d\texpected: %d\tfound: %d" %(i, ground_truth_y[i], y_hat[i]))
        print("Initial Loss: %1.5f" %initial_loss)
        ## End Test Code ##

        # Create list to save change value for each weight
        change_list = []

        # For each weight value check which produces better loss, weight +/- step_size
        # Go through each hidden layer
        for i_layer in range(net.n):
            # Add list to change_list
            change_list.append([])

            # Go through each neuron in the layer
            for i_neuron in range(net.hidden_layers[i_layer].m):
                # Add list to layer's list in change_list
                change_list[i_layer].append([])
                
                # Once again iterate through the weights changing them by the step_size
                # both positive and negative. Choosing the value which creates the smaller
                # loss.
                for i_weight in range(len(net.hidden_layers[i_layer].neurons[i_neuron].weights)):
                    # Remember initial
                    initial_value = net.hidden_layers[i_layer].neurons[i_neuron].weights[i_weight]

                    # Check adding step_size
                    net.hidden_layers[i_layer].neurons[i_neuron].weights[i_weight] += step_size
                    test_loss_add = loss_function.error(net.forward(train_data), ground_truth_y)
                    
                    # Reset weight
                    net.hidden_layers[i_layer].neurons[i_neuron].weights[i_weight] = initial_value

                    # Check subtracting step_size
                    net.hidden_layers[i_layer].neurons[i_neuron].weights[i_weight] -= step_size
                    test_loss_sub = loss_function.error(net.forward(train_data), ground_truth_y)

                    # Reset weight
                    net.hidden_layers[i_layer].neurons[i_neuron].weights[i_weight] = initial_value

                    # Check which one was better
                    if test_loss_add < test_loss_sub:
                        change_list[i_layer][i_neuron].append(step_size)
                    elif test_loss_sub < test_loss_add:
                        change_list[i_layer][i_neuron].append(-step_size)
                    else:
                        change_list[i_layer][i_neuron].append(0)

        # Apply changes found in perturb step
        # For each weight value check which produces better loss, weight +/- step_size
        # Go through each hidden layer
        for i_layer in range(net.n):
            # Go through each neuron in the layer
            for i_neuron in range(net.hidden_layers[i_layer].m):
                # Once again iterate through the weights changing them by the step_size
                # both positive and negative. Choosing the value which creates the smaller
                # loss.
                for i_weight in range(len(net.hidden_layers[i_layer].neurons[i_neuron].weights)):
                    net.hidden_layers[i_layer].neurons[i_neuron].weights[i_weight] += change_list[i_layer][i_neuron][i_weight]

        # Check if convergence has been found
        if prev_loss == initial_loss:
            print("----- Convergence Found -----")
            break

        # Update prev_loss
        prev_loss = initial_loss

if __name__ == "__main__":
    """
    If this file is being run individually test the implementation of training functions.
    """

    test_inputs = np.array([[2], [1]])

    ## Test perturb_train Function Implementation ##
    print("Testing perturb_train")
    test_network = LinearNeuralNetwork(2, 1, 2, 1, sigmoid())
    perturb_train(test_network, test_inputs, [1], MeanSquaredError(), epochs=1000, step_size=0.001)
