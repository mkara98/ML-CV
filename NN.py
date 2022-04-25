import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def relu(z):
    a = np.maximum(0, z)
    return a


def initialize_params(layer_sizes):
    params = {}
    for i in range(1, len(layer_sizes)):
        params['W' + str(i)] = np.random.randn(layer_sizes[i], layer_sizes[i - 1]) * 0.01
        params['B' + str(i)] = np.random.randn(layer_sizes[i], 1) * 0.01
    return params


def forward_propagation(X_train, params):
    layers = len(params) // 2
    values = {}
    for i in range(1, layers + 1):
        if i == 1:
            values['Z' + str(i)] = np.dot(params['W' + str(i)], X_train) + params['B' + str(i)]
            print("Yazdır")
            print(np.dot(params['W' + str(i)], X_train).shape)
            print(params['B' + str(i)].shape)
            values['A' + str(i)] = relu(values['Z' + str(i)])
        else:
            values['Z' + str(i)] = np.dot(params['W' + str(i)], values['A' + str(i - 1)]) + params['B' + str(i)]
            if i == layers:
                values['A' + str(i)] = values['Z' + str(i)]
            else:
                values['A' + str(i)] = relu(values['Z' + str(i)])
    return values


def compute_cost(values, Y_train):
    layers = len(values) // 2
    Y_pred = values['A' + str(layers)]
    cost = 1 / (2 * len(Y_train)) * np.sum(np.square(Y_pred - Y_train))
    return cost


def backward_propagation(params, values, X_train, Y_train):
    layers = len(params) // 2
    m = len(Y_train)
    grads = {}
    for i in range(layers, 0, -1):
        if i == layers:
            dA = 1 / m * (values['A' + str(i)] - Y_train)
            dZ = dA
        else:
            dA = np.dot(params['W' + str(i + 1)].T, dZ)
            dZ = np.multiply(dA, np.where(values['A' + str(i)] >= 0, 1, 0))
        if i == 1:
            grads['W' + str(i)] = 1 / m * np.dot(dZ, X_train.T)
            grads['B' + str(i)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        else:
            grads['W' + str(i)] = 1 / m * np.dot(dZ, values['A' + str(i - 1)].T)
            grads['B' + str(i)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    print(grads['W1'].shape)
    print(grads['B1'].shape)
    return grads


def update_params(params, grads, learning_rate):
    layers = len(params) // 2
    params_updated = {}
    for i in range(1, layers + 1):
        params_updated['W' + str(i)] = params['W' + str(i)] - learning_rate * grads['W' + str(i)]
        params_updated['B' + str(i)] = params['B' + str(i)] - learning_rate * grads['B' + str(i)]
    return params_updated


def model(X_train, Y_train, layer_sizes, num_iters, learning_rate):
    params = initialize_params(layer_sizes)
    for i in range(num_iters):
        values = forward_propagation(X_train.T, params)
        cost = compute_cost(values, Y_train.T)
        grads = backward_propagation(params, values, X_train.T, Y_train.T)
        params = update_params(params, grads, learning_rate)
        # print('Cost at iteration ' + str(i + 1) + ' = ' + str(cost) + '\n')
    return params


def compute_accuracy(X_train, X_test, Y_train, Y_test, params):
    values_train = forward_propagation(X_train.T, params)
    values_test = forward_propagation(X_test.T, params)
    train_acc = mean_squared_error(Y_train, values_train['A' + str(len(layer_sizes) - 1)].T)
    test_acc = mean_squared_error(Y_test, values_test['A' + str(len(layer_sizes) - 1)].T)
    return train_acc, test_acc


def predict(X, params):
    values = forward_propagation(X.T, params)
    predictions = values['A' + str(len(values) // 2)].T
    return predictions


data = load_boston()  # load dataset
X, Y = data["data"], data["target"]  # separate data into input and output features
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2)  # split data into train and test sets in 80-20 ratio
layer_sizes = [13, 1]  # set layer sizes, do not change the size of the first and last layer
num_iters = 1  # set number of iterations over the training set(also known as epochs in batch gradient descent context)
learning_rate = 0.03  # set learning rate for gradient descent
params = model(X_train, Y_train, layer_sizes, num_iters, learning_rate)  # train the model
train_acc, test_acc = compute_accuracy(X_train, X_test, Y_train, Y_test, params)  # get training and test accuracy
print('Root Mean Squared Error on Training Data = ' + str(train_acc))
print('Root Mean Squared Error on Test Data = ' + str(test_acc))
