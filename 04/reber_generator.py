#!/usr/bin/env python3

import sys
import random

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

def generatorProzess(queue,stop, lenght):
    lenght = lenght.value
    while(stop != 1):
        string = cell_1().appendString("B")
        if(len(string) == lenght):
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

def generate(seq_len, anzahl, filename):
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

if __name__ == "__main__":
    main()
