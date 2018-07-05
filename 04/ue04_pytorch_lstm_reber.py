import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from reber_gen import get_n_examples
import numpy as np
import pickle
from reber_gen import ReberDataset

mini_batch_sizes = [1, 5, 10, 50, 100]
EMBEDDING_DIM = 7
HIDDEN_DIM = 12
MINI_BATCH_SIZE = 5
WORD_SIZE = 30
LEARNING_RATE = 0.1
NUM_TEST = 500
MOMENTUM = 0.9
OPTIMIZER = ['SGD', 'RMSProp', 'AdaDelta'][2]

def print_fancy(*args):
    print('-'*100)
    for a in args:
        print(a)
        print('*'*100)
    print('-'*100)

# torch.manual_seed(1)
def prepare_sequence(seq):
    return torch.tensor(seq, dtype=torch.float)

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

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, WORD_SIZE, self.hidden_dim),
                torch.zeros(1, WORD_SIZE, self.hidden_dim))

    def forward(self, sentence):
        lstm_out, self.hidden = self.lstm(sentence, self.hidden)
        activation_1 = F.sigmoid(lstm_out)
        tag_space = self.hidden2tag(activation_1)
        activation = F.sigmoid(tag_space)
        return activation


train_dataset = ReberDataset(10000-NUM_TEST, WORD_SIZE, test=False)
test_dataset = ReberDataset(NUM_TEST, WORD_SIZE, test=True)

for mbs in mini_batch_sizes:
    
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, 7, 7)
    if OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    elif OPTIMIZER == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER == 'AdaDelta':
        optimizer = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)
    else:
        OPTIMIZER = 'Adam'
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=mbs,
                                                shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=mbs,
                                                shuffle=False, num_workers=2)

    loss_function = nn.MSELoss()

    print('-|-'*100)
    print('Optimizer {}, Learning Rate {}, Batch Size {}'.format(OPTIMIZER, LEARNING_RATE, mbs))

    for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in trainloader:
            model.zero_grad()

            model.hidden = model.init_hidden()

            sentence_in = prepare_sequence(sentence)
            targets = prepare_sequence(tags)

            tag_scores = model(sentence_in)

            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
        print('Epoch {}'.format(epoch))
        print('loss -> ', loss.item())

    # See what the scores are after training
    print('Test Data:')
    print_procentage_correctly_classified(model, testloader)
    print('Training Data')
    print_procentage_correctly_classified(model, trainloader)
    print('*'*100)