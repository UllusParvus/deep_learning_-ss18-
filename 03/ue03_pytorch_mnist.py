from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # define all the components that will be used in the NN (these can be reused)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2, padding=0)
        self.drop2D = nn.Dropout2d(p=0.25, inplace=False)
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # define the acutal network
        in_size = x.size(0)  # get the batch size

        # chain function together to form the layers
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = self.drop2D(x)
        x = x.view(in_size, -1)  # flatten data, -1 is inferred from the other dimensions
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # 2 Convolutional and 2 pooling layers
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.pool1 = nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.pool2 = nn.MaxPool2d(2)
#
#     def forward(self, input):
#         in_size = input.size(0)
#         x = self.conv1(input)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.pool2(x)
#         return x


batch_size = 100

train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)

# batch the data for the training and test datasets
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

print(train_loader.__len__()*train_loader.batch_size, 'train samples')
print(test_loader.__len__()*test_loader.batch_size, 'test samples\n')

net = Net()
net.cuda()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's weights size
print(params[0][0, 0])  # conv1's weights for the first filter's kernel


def train(epoch):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data.cuda() loads the data on the GPU, which increases performance
        data, target = Variable(data.cuda()), Variable(target.cuda())

        optimizer.zero_grad()  # necessary for new sum of gradients
        output = net(data)  # call the forward() function (forward pass of network)
        loss = F.nll_loss(output, target)  # use negative log likelihood to determine loss
        loss.backward()  # backward pass of network (calculate sum of gradients for graph)
        optimizer.step()  # perform model perameter update (update weights)


def test(epoch):
    net.eval()  # set the model in "testing mode"
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        data, target = Variable(data.cuda(), volatile=True), Variable(
            target.cuda())  # volatile=True, since the test data should not be used to train... cancel backpropagation
        output = net(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[
            0]  # fsize_average=False to sum, instead of average losses
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(
            target.data.view_as(pred)).cpu().sum()  # to operate on variables they need to be on the CPU again

    test_dat_len = len(test_loader.dataset)
    test_loss /= test_dat_len

    # print the test accuracy
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, test_dat_len, 100. * correct / test_dat_len))


epochs = 10

if __name__ == '__main__':
    for epoch in range(1, epochs):
        train(epoch)
        test(epoch)
