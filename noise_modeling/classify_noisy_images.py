import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#############################################################################
# 1. Get training and test data sets
#############################################################################
transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor()])
batch_size = 10
trainset = torchvision.datasets.MNIST(
    './data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)

testset = (torchvision.datasets.MNIST(
    './data', train=False, download=True, transform=transform))
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True)

data_path = './noisy_images/gaussian'
gaussian_testset = torchvision.datasets.ImageFolder(
    root=data_path, transform=transform)
gaussian_testloader = torch.utils.data.DataLoader(
    gaussian_testset, batch_size=batch_size, shuffle=True)

data_path = './noisy_images/pgd_attacked'
pgd_testset = torchvision.datasets.ImageFolder(
    root=data_path, transform=transform)
pgd_testloader = torch.utils.data.DataLoader(
    pgd_testset, batch_size=batch_size, shuffle=True)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
PATH = '../models/MNIST_net.pth'

#############################################################################
# 2. Define CNN, loss, and optimizer
#############################################################################
class MNIST_net(nn.Module):
    def __init__(self):
        super(MNIST_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 4 * 4, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x

#############################################################################
# 4. Testing with the test data
#############################################################################
test_net = MNIST_net()
test_net.load_state_dict(torch.load(PATH))
correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = test_net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        c = (predicted == labels).squeeze()
        for i in range(batch_size):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
print('Accuracy of the network on the GAN-generated test images: %d %%'
      % (100 * correct / total))
for i in range(10):
    print('Accuracy of %s %2d %%'
          % (classes[i], 100 * class_correct[i] / class_total[i]))

correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in gaussian_testloader:
        images, labels = data
        outputs = test_net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        c = (predicted == labels).squeeze()
        for i in range(batch_size):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
print('Accuracy of the network on the Gaussian test images: %d %%'
      % (100 * correct / total))
for i in range(10):
    print('Accuracy of %s %2d %%'
          % (classes[i], 100 * class_correct[i] / class_total[i]))

correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in pgd_testloader:
        images, labels = data
        outputs = test_net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        c = (predicted == labels).squeeze()
        for i in range(batch_size):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
print('Accuracy of the network on the PGD test images: %d %%'
      % (100 * correct / total))
for i in range(10):
    print('Accuracy of %s %2d %%'
          % (classes[i], 100 * class_correct[i] / class_total[i]))