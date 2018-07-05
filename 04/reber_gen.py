#!/usr/bin/python

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os.path
from reber_generator import generate

chars = 'BTSXPVE'

graph = [[(1, 5), ('T', 'P')], [(1, 2), ('S', 'X')], \
         [(3, 5), ('S', 'X')], [(6,), ('E')], \
         [(3, 2), ('V', 'P')], [(4, 5), ('V', 'T')]]


def in_grammar(word):
    if word[0] != 'B':
        return False
    node = 0
    for c in word[1:]:
        transitions = graph[node]
        try:
            node = transitions[0][transitions[1].index(c)]
        except ValueError:  # using exceptions for flow control in python is common
            return False
    return True


def sequenceToWord(sequence):
    """
    converts a sequence (one-hot) in a reber string
    """
    reberString = ''
    for s in sequence:
        index = np.where(s == 1.)[0][0]
        reberString += chars[index]
    return reberString


def generateSequences(minLength):
    while True:
        inchars = ['B']
        node = 0
        outchars = []
        while node != 6:
            transitions = graph[node]
            i = np.random.randint(0, len(transitions[0]))
            inchars.append(transitions[1][i])
            outchars.append(transitions[1])
            node = transitions[0][i]
        if len(inchars) == minLength:
            return inchars, outchars


def get_one_example(minLength):
    inchars, outchars = generateSequences(minLength)
    inseq = []
    outseq = []
    for i, o in zip(inchars, outchars):
        inpt = np.zeros(7)
        inpt[chars.find(i)] = 1.
        outpt = np.zeros(7)
        for oo in o:
            outpt[chars.find(oo)] = 1.
        inseq.append(inpt)
        outseq.append(outpt)
    inseq.append(get_char_one_hot(('E',))[0])
    outseq.append(get_char_one_hot(())[0])
    return inseq, outseq


def get_char_one_hot(char):
    char_oh = np.zeros(7)
    for c in char:
        char_oh[chars.find(c)] = 1.
    return [char_oh]


def get_n_examples(n, minLength=10):
    examples = []
    for i in range(n):
        examples.append(get_one_example(minLength))
    return examples

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


class ReberDataset(Dataset):
    def __init__(self, num, length, test):
        self.test = test

        fname = './data10000_length_{}'.format(length)
        if not os.path.isfile(fname):
            generate(length, 10000, fname)
        data = load_data(fname)
        length_test_data = 500
        self.test_data = data[:length_test_data]
        self.training_data = data[length_test_data:]

    def __getitem__(self, index):
        if self.test:
            return torch.tensor(self.test_data[index][0]), torch.tensor(self.test_data[index][1])
        else:
            return torch.tensor(self.training_data[index][0]), torch.tensor(self.training_data[index][1])
            
    def __len__(self):
        if self.test:
            return len(self.test_data)
        else:
            return len(self.training_data)

if __name__ == '__main__':
    train_data = get_n_examples(1000, 7)
    print(train_data[0][0])
    print(train_data[0][1])