import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def mean_squared_error(predictions, actual):
    return 1/predictions.shape[0] * ((actual.T - predictions) ** 2).sum(axis=0)


# Trainingsdaten
training_input = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])

training_output = np.array([[0, 0, 0],
                            [0, 1, 1],
                            [0, 1, 1],
                            [1, 1, 0]])

if __name__ == '__main__':
    input_set = np.array([[0, 0]])

    weights_h = np.array(np.random.uniform(-0.1, 0.1, (2, 4)))
    weights_o = np.array(np.random.uniform(-0.1, 0.1, (4, 3)))

    bias_h = np.array(np.random.uniform(-0.1, 0.1, (1, 4)))
    bias_o = np.array(np.random.uniform(-0.1, 0.1, (1, 3)))

    hidden_layer = sigmoid(np.dot(weights_h.T, input_set.T) + bias_h.T)
    output_layer = sigmoid(np.dot(weights_o.T, hidden_layer) + bias_o.T)

    for i in range(0, 1000):
        for j in range(0, training_input.shape[0]):
            input_set = np.array([training_input[j]])
            actual = np.array([training_output[j]])
            hidden_layer = sigmoid(np.dot(weights_h.T, input_set.T) + bias_h.T)
            predictions = sigmoid(np.dot(weights_o.T, hidden_layer) + bias_o.T)

            mse = mean_squared_error(predictions, actual)

            print('### Schritt ' + str(i+1) + ' ###')
            print('Vorhersage für ' + str(input_set) + ' --> ' + str(output_layer))
            print('MSE --> ' + str(mse))

            Delta_L_by_A = 2*predictions - 2*actual




    '''
    print(training_input.shape)
    print('Input: ' + str(input_set[0]))
    print('NN -> AND: ' + str(output_layer[0][0]))
    print('NN -> OR: ' + str(output_layer[1][0]))
    print('NN -> XOR: ' + str(output_layer[2][0]))
    '''