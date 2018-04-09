import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    input_set = np.array([[0, 0]])

    weights_h = np.array([[20, 20, 20, -20], [20, 20, 20, -20]])
    weights_o = np.array([[30, 1, 1], [1, 30, 1], [1, 1, 20], [1, 1, 20]])

    bias_h = np.array([[-30, -10, -10, 30]])
    bias_o = np.array([[-15, -15, -30]])

    hidden_layer = sigmoid(np.dot(weights_h.T, input_set.T) + bias_h.T)
    output_layer = sigmoid(np.dot(weights_o.T, hidden_layer) + bias_o.T)

    print('Input: ' + str(input_set[0]))
    print('NN -> AND: ' + str(output_layer[0][0]))
    print('NN -> OR: ' + str(output_layer[1][0]))
    print('NN -> XOR: ' + str(output_layer[2][0]))