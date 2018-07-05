#!/usr/bin/env python3

import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os.path

from multiprocessing import Process, Queue, Value

#~ import rstr

class cell:
    def __init__(self):
        self.nextcells = []

    def appendString(self, string):
        index = random.randint(0, len(self.nextcells)-1)
        string += self.nextcells[index][0]
        return self.nextcells[index][1]().appendString(string)

class cell_1(cell):
    def __init__(self):
        self.nextcells = [("T", cell_2), ("P",cell_4)]

class cell_2(cell):
    def __init__(self):
        self.nextcells = [("S", cell_2), ("X",cell_3)]
class cell_3(cell):
    def __init__(self):
        self.nextcells = [("X", cell_4), ("S",cell_6)]

class cell_4(cell):
    def __init__(self):
        self.nextcells = [("T", cell_4), ("V",cell_5)]

class cell_5(cell):
    def __init__(self):
        self.nextcells = [("P", cell_3), ("V",cell_6)]

class cell_6(cell):
    def appendString(self, string):
        return string + "E"

def generatorProzess(queue, stop, length):
    length = length.value
    while(stop != 1):
        string = cell_1().appendString("B")
        if(len(string) == length):
            queue.put(string)

def main():
    if(len(sys.argv) < 4):
        print("Use " + sys.argv[0] + " [Sequenzlaenge] [anzahl] [outfilename]")
        exit()
    seqlen = int(sys.argv[1])
    anzahl = int(sys.argv[2])
    filename = sys.argv[3]

    if(seqlen < 5):
        print("Seqlen to small")
        exit()

    stop = Value('d', 0)
    lenght = Value('d', seqlen)
    threadAnz = 8
    queues = []
    processes = []
    for i in range(0, threadAnz):
        queue = Queue(100)
        queues.append(queue)
        p = Process(target=generatorProzess, args=(queue,stop,lenght))
        processes.append(p)
        p.start()

    output = set()
    while(1):
        for queue in queues:
            string = queue.get()
            output.add(string)
            print("hit", len(output))
            if(len(output) >= anzahl):
                break
        if(len(output) >= anzahl):
            print("break")
            for p in processes:
                p.terminate()
            break

    outputfile = open(filename, 'w')
    for o in output:
        outputfile.write(str(o) + "\n")
    #~ outputfile.close()

def generate(seqlen, anzahl, filename):
    if(seqlen < 5):
        print("Seqlen to small")
        exit()

    stop = Value('d', 0)
    length = Value('d', seqlen)
    threadAnz = 8
    queues = []
    processes = []
    for i in range(0, threadAnz):
        queue = Queue(100)
        queues.append(queue)
        p = Process(target=generatorProzess, args=(queue,stop,length))
        processes.append(p)
        p.start()

    output = set()
    while(1):
        for queue in queues:
            string = queue.get()
            output.add(string)
            print("hit", len(output))
            if(len(output) >= anzahl):
                break
        if(len(output) >= anzahl):
            print("break")
            for p in processes:
                p.terminate()
            break

    outputfile = open(filename, 'w')
    for o in output:
        outputfile.write(str(o) + "\n")
    #~ outputfile.close()


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


if __name__ == "__main__":
    main()
