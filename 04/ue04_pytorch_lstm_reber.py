import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from reber_gen import get_n_examples

def print_fancy(*args):
    print('-'*100)
    for a in args:
        print(a)
        print('*'*100)
    print('-'*100)

torch.manual_seed(1)
def prepare_sequence(seq):
    return torch.tensor([seq], dtype=torch.float)

EMBEDDING_DIM = 7
HIDDEN_DIM = 12
MINI_BATCH_SIZE = 1
WORD_SIZE = 13
LEARNING_RATE = 0.1

training_data = get_n_examples(1000, WORD_SIZE)

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
        tag_space = self.hidden2tag(lstm_out)
        activation = F.sigmoid(tag_space)
        # tag_scores = F.log_softmax(tag_space, dim=1)
        return activation

# model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, 7, 7)
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0])
    tag_scores = model(inputs)

for epoch in range(30):  # again, normally you would NOT do 300 epochs, it is toy data
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
    print('loss -> ', loss.item())

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0])
    label = prepare_sequence(training_data[0][1])
    tag_scores = model(inputs)
    print('Inputs: ', inputs)
    print('Label', label)
    print('Prediction', tag_scores)