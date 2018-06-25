import numpy as np
from torch.utils.data import Dataset

alphabet = list('btpxsev'.upper())

reber = [
    {'B': 1},
    {'T': 6, 'P': 2},
    {'T': 2, 'V': 3},
    {'P': 5, 'V': 4},
    {'E': 7},
    {'X': 2, 'S': 4},
    {'S': 6, 'X': 5},
    {}
]

def goto_next_state(state, letter):
    if letter.upper() in reber[state]:
        return reber[state][letter.upper()]
    return state

def get_next_possible_letters(state):
    return list(reber[state].keys())

def get_last_possible_letters(word):
    state = 0
    for l in word:
        state = goto_next_state(state, l)
    return get_next_possible_letters(state)

def generate_word(length):
    word = np.random.choice(alphabet, length)
    letters = get_last_possible_letters(word)
    # print(word, letters)
    return word, letters

def generate_data_set(num, length):
    x_data = []
    y_data = []
    for i in range(num):
        x, y = generate_word(length)
        x_data.append(x)
        y_data.append(y)

    trans_dict = {l: n for n, l in enumerate(alphabet)}
    #print(test_data)
    x_data = [[[1 if i == trans_dict[a] else 0 for i in range(len(alphabet))] for a in ele] for ele in x_data]
    y_data = [[[1 if i == trans_dict[a] else 0 for i in range(len(alphabet))] for a in ele] for ele in y_data]
    y_data = [np.vstack(ele).sum(axis=0).tolist() for ele in y_data]
    return list(zip(x_data, y_data))


class ReberDataset(Dataset):
    def __init__(self, num, length):
        print(num, length)
        self.data = generate_data_set(num, length)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    print(generate_data_set(100, 7))