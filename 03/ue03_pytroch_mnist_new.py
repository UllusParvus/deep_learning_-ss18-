# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler
import os
import matplotlib.pyplot as plt
import torch.optim as optim

# Konfiguration
momentum = 0.9
learning_rate = 0.001
mini_batch_size = 10
mini_batch_sizes = [1, 10, 50, 100]
optimizer_type = 'rms_prop'

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,  256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    for batch_size in mini_batch_sizes:
        print("Optimizer {}, batch_size {}, learning_rate {}, momentum {}".format(optimizer_type, batch_size, learning_rate, momentum))
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)
        train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
        test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)

        classes = np.arange(10)  # classes from 0-9

        # get some random training images

        dataiter = iter(trainloader)
        images, labels = dataiter.next()

        # show images
        imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join('%5s' % classes[labels[j]] for j in range(1)))

        net = Net()

        criterion = nn.CrossEntropyLoss()

        if optimizer_type == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
        elif optimizer_type == 'rms_prop':
            optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, momentum=momentum)
        elif optimizer_type == 'ada_delta':
            optimizer = optim.Adadelta(net.parameters(), lr=learning_rate, rho=0.9, eps=1e-06, weight_decay=0)
        else:
            # default sgd
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

        for epoch in range(2):

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

        dataiter = iter(testloader)
        images, labels = dataiter.next()

        # print images
        imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(1)))

        outputs = net(images)

        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                      for j in range(1)))

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(1):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(10):
            if class_total[i] == 0:
                class_total[i] = 0.00001
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))


        dataiter._shutdown_workers()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(device)