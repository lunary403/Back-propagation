from random import random
from random import seed
from math import exp
from preprocessing import *
import numpy as np

df = read_file("Dry_Bean_Dataset.xlsx")
df["MinorAxisLength"] = fill_null_values(df, "MinorAxisLength")

X_train, y_train, X_test, y_test =train_test_split(df)


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs, n_hidden_layers, bias=True):
    seed(42)
    network = list()

    # Hidden Layers
    for _ in range(n_hidden_layers):
        if bias:
            hidden_layer = [{'weights': [random() for _ in range(n_inputs + 1)]} for _ in range(n_hidden)]
        else:
            hidden_layer = [{'weights': [random() for _ in range(n_inputs)]} for _ in range(n_hidden)]
        network.append(hidden_layer)
        n_inputs = n_hidden  # Update n_inputs for the next layer

    # Output Layer
    if bias:
        output_layer = [{'weights': [random() for _ in range(n_hidden + 1)]} for _ in range(n_outputs)]
    else:
        output_layer = [{'weights': [random() for _ in range(n_hidden)]} for _ in range(n_outputs)]
    network.append(output_layer)

    return network






# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation




# Transfer neuron activation (sigmoid)
def transfer_sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))

def transfer_tanh(activation):
    return (np.exp(activation) - np.exp(-activation))/(np.exp(activation) + np.exp(-activation))



# Forward propagate input to a network output
def forward_propagate_sigmoid(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer_sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def forward_propagate_tanh(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer_tanh(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs



# Calculate the derivative of an neuron output (sigmoid deravitive)
def transfer_derivative_sigmoid(output):
    return output * (1.0 - output)


def transfer_derivative_tanh(output):
    return 1 - output*output


# Backpropagate error and store in neurons

def backward_propagate_error_sigmoid(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()

        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += neuron['weights'][j] * neuron['delta']
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])

        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative_sigmoid(neuron['output'])

def backward_propagate_error_tanh(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()

        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += neuron['weights'][j] * neuron['delta']
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])

        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative_tanh(neuron['output'])

def update_weights(network, row, l_rate, bias=True):
    for i in range(len(network)):
        inputs = row[:-1] if not bias else row  # Exclude the last element if bias is False
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            if bias:
                neuron['weights'][-1] -= l_rate * neuron['delta']






def train_network_sigmoid(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        sum_error = 0

        for row in train:
            # Forward propagation
            outputs = forward_propagate_sigmoid(network, row)

            # Convert network output to binary predictions (0 or 1)
            predicted = [1 if output > 0.5 else 0 for output in outputs]

            # Convert the true label to binary format
            expected = [0] * n_outputs
            expected[int(row[-1])] = 1

            # Update confusion matrix
            for i in range(len(expected)):
                if expected[i] == 1 and predicted[i] == 1:
                    true_positive += 1
                elif expected[i] == 0 and predicted[i] == 0:
                    true_negative += 1
                elif expected[i] == 0 and predicted[i] == 1:
                    false_positive += 1
                elif expected[i] == 1 and predicted[i] == 0:
                    false_negative += 1

            # Calculate error for the current row
            sum_error += sum((expected[i] - outputs[i]) ** 2 for i in range(len(expected)))

            # Backward propagation
            backward_propagate_error_sigmoid(network, expected)

            # Update weights
            update_weights(network, row, l_rate)

        # Output confusion matrix components at the end
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (
            epoch, l_rate, sum_error))

    # Output final confusion matrix components
    print('Final Confusion Matrix:')
    print('TP:', true_positive)
    print('TN:', true_negative)
    print('FP:', false_positive)
    print('FN:', false_negative)
    overall_acc = (true_positive + true_negative)/(true_positive+true_negative+false_positive + false_negative)
    print("overall accuracy: ", overall_acc)

def train_network_tanh(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        sum_error = 0

        for row in train:
            # Forward propagation
            outputs = forward_propagate_tanh(network, row)

            # Convert network output to binary predictions (0 or 1)
            predicted = [1 if output > 0.5 else 0 for output in outputs]

            # Convert the true label to binary format
            expected = [0] * n_outputs
            expected[int(row[-1])] = 1

            # Update confusion matrix
            for i in range(len(expected)):
                if expected[i] == 1 and predicted[i] == 1:
                    true_positive += 1
                elif expected[i] == 0 and predicted[i] == 0:
                    true_negative += 1
                elif expected[i] == 0 and predicted[i] == 1:
                    false_positive += 1
                elif expected[i] == 1 and predicted[i] == 0:
                    false_negative += 1

            # Calculate error for the current row
            sum_error += sum((expected[i] - outputs[i]) ** 2 for i in range(len(expected)))

            # Backward propagation
            backward_propagate_error_tanh(network, expected)

            # Update weights
            update_weights(network, row, l_rate)

        # Output confusion matrix components at the end
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (
            epoch, l_rate, sum_error))

    # Output final confusion matrix components
    print('Final Confusion Matrix:')
    print('TP:', true_positive)
    print('TN:', true_negative)
    print('FP:', false_positive)
    print('FN:', false_negative)
    overall_acc = (true_positive + true_negative)/(true_positive+true_negative+false_positive + false_negative)
    print("overall accuracy: ", overall_acc)
