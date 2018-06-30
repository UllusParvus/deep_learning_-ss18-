import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from reber_gen import get_n_examples
import numpy as np
import pickle

alphabet = 'BTSXPVE'
trans_dict = {a:np.eye(7)[i] for i, a in enumerate(alphabet)}

graph = [[(1, 5), ('T', 'P')], [(1, 2), ('S', 'X')], \
         [(3, 5), ('S', 'X')], [(6,), ('E')], \
         [(3, 2), ('V', 'P')], [(4, 5), ('V', 'T')]]

def get_y_label(letter, state):
    next_states, letters = graph[state]
    if letter in letters:
        idx = letters.index(letter)
        next_state = next_states[idx]

        if next_state == 6: # End State:
            return -1, [0.0]*len(alphabet)
        
        _, new_letters = graph[next_state]
        label = np.vstack([trans_dict[a] for a in new_letters]).sum(axis=0).tolist()
        return next_state, label
    else:
        return -1, [0.0]*len(alphabet)

def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        for word in file:
            word = word.rstrip()
            encoded_x = []
            encoded_y = []
            state = 0
            for letter in word:
                state, label = get_y_label(letter, state)
                encoded_x.append(trans_dict[letter])
                encoded_y.append(label)
            data.append((encoded_x, encoded_y))
    return data

def print_fancy(*args):
    print('-'*100)
    for a in args:
        print(a)
        print('*'*100)
    print('-'*100)

# torch.manual_seed(1)
def prepare_sequence(seq):
    return torch.tensor([seq], dtype=torch.float)

def get_num_correct_classified(x, y):
    num = 0
    for inp, label in zip(x, y):
        if is_correctly_classified(inp, label):
            num += 1
    return num

def is_correctly_classified(inp, label):
    return np.all((np.round(inp) - label) == 0)

def print_procentage_correctly_classified(model, data):
    max_num = 0
    correct = 0
    with torch.no_grad():
        for x, y in data:
            inputs = prepare_sequence(x)
            label = prepare_sequence(y)
            tag_scores = model(inputs)
            max_num += WORD_SIZE
            correct += get_num_correct_classified(tag_scores[0], label[0])
    print('prozent', correct/max_num)
    return correct/max_num

EMBEDDING_DIM = 7
HIDDEN_DIM = 12
MINI_BATCH_SIZE = 1
WORD_SIZE = 30
LEARNING_RATE = 0.0001

# NUM_TRAINING_EXAMPLES = 1000
#
# import os.path
# file_name = './cache{}_length{}.data'.format(NUM_TRAINING_EXAMPLES, WORD_SIZE)
# if os.path.isfile(file_name):
#     print('Found {} loading...'.format(file_name))
#     with open(file_name, 'rb') as file:
#         training_data = pickle.loads(file.read())
#         print('Done loading')
# else:
#     print('generating {} words with length {}'.format(NUM_TRAINING_EXAMPLES, WORD_SIZE))
#     training_data = get_n_examples(NUM_TRAINING_EXAMPLES, WORD_SIZE)
#     print('generation finished')
#     with open(file_name, 'wb') as file:
#         print('saving...')
#         file.write(pickle.dumps(training_data))
#         print('finished saving {}'.format(file_name))

# test_data = get_n_examples(500, WORD_SIZE)

data = load_data('./data10000_length_16_jodani')
length_test_data = 500
test_data = data[:length_test_data]
training_data = data[length_test_data:]

print('finished loading {} examples'.format(len(data)))

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, MINI_BATCH_SIZE, self.hidden_dim),
                torch.zeros(1, MINI_BATCH_SIZE, self.hidden_dim))

    def forward(self, sentence):
        lstm_out, self.hidden = self.lstm(sentence, self.hidden)
        activation_1 = F.sigmoid(lstm_out)
        tag_space = self.hidden2tag(activation_1)
        activation = F.sigmoid(tag_space)
        return activation


# model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, 7, 7)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0])
    tag_scores = model(inputs)

for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # print('sentence -> ' + str(sentence) + ', tag -> ' + str(tags))
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # for letter, probs in zip(sentence, tags):
        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        #print(word_to_ix)
        sentence_in = prepare_sequence(sentence)
        targets = prepare_sequence(tags)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)
        # print('tag_score', tag_scores)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        # print_fancy(epoch, sentence_in, targets, tag_scores, loss)
        loss.backward()
        optimizer.step()
    print('Epoch {}'.format(epoch))
    print('loss -> ', loss.item())

    # See what the scores are after training
    print('Test Data:')
    print_procentage_correctly_classified(model, test_data)
    print('Training Data')
    print_procentage_correctly_classified(model, training_data)
    print('*'*100)