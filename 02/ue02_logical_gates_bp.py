import numpy as np

LEARNING_RATE = 0.1

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
        print('### Schritt ' + str(i + 1) + ' ###')
        for j in range(0, training_input.shape[0]):
            input_set = np.array([training_input[j]])
            actual = np.array([training_output[j]])

            L1 = np.dot(weights_h.T, input_set.T) + bias_h.T
            A1 = sigmoid(L1)

            L2 = np.dot(weights_o.T, A1) + bias_o.T
            A2 = sigmoid(L2)

            mse = mean_squared_error(A2, actual)

            print('Vorhersage für ' + str(input_set) + ' --> ' + str(output_layer))
            print('MSE --> ' + str(mse))

            # output layer
            Delta_L_by_A2 = 2*A2 - 2*actual.T
            Delta_L_by_L2 = Delta_L_by_A2 * sigmoid_derivative(L2)
            Delta_L_by_B2 = Delta_L_by_L2
            Delta_L_by_W2 = np.dot(Delta_L_by_B2, A1.T)

            # hidden layer
            Delta_L_by_A1 = np.dot(weights_o, Delta_L_by_L2)
            Delta_L_by_L1 = Delta_L_by_A1 * sigmoid_derivative(L1)
            Delta_L_by_B1 = Delta_L_by_L1
            Delta_L_by_W1 = np.dot(Delta_L_by_B1, input_set)

            weights_h = weights_h - LEARNING_RATE * Delta_L_by_W1.T
            weights_o = weights_o - LEARNING_RATE * Delta_L_by_W2.T

            bias_h = bias_h - LEARNING_RATE * Delta_L_by_B1.T
            bias_o = bias_o - LEARNING_RATE * Delta_L_by_B2.T


    '''
    print(training_input.shape)
    print('Input: ' + str(input_set[0]))
    print('NN -> AND: ' + str(output_layer[0][0]))
    print('NN -> OR: ' + str(output_layer[1][0]))
    print('NN -> XOR: ' + str(output_layer[2][0]))
    '''