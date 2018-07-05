import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from reber_gen import get_n_examples
import numpy as np
import pickle
from reber_gen import ReberDataset

def print_fancy(*args):
    print('-'*100)
    for a in args:
        print(a)
        print('*'*100)
    print('-'*100)

# torch.manual_seed(1)
def prepare_sequence(seq):
    print('seq', len(seq))
    print('seq', len(seq[0]))
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

EMBEDDING_DIM = 7
HIDDEN_DIM = 12
MINI_BATCH_SIZE = 5
WORD_SIZE = 30
LEARNING_RATE = 0.0001
NUM_TEST = 500


train_dataset = ReberDataset(10000-NUM_TEST, WORD_SIZE, test=False)
test_dataset = ReberDataset(NUM_TEST, WORD_SIZE, test=True)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=MINI_BATCH_SIZE,
                                            shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=MINI_BATCH_SIZE,
                                            shuffle=False, num_workers=2)

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim) # batch_first=True

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
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

print('Optimizer Adam, Learning Rate {}, Batch Size 1'.format(LEARNING_RATE))

for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in trainloader:
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
    print_procentage_correctly_classified(model, testloader)
    print('Training Data')
    print_procentage_correctly_classified(model, trainloader)
    print('*'*100)