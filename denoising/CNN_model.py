import torch.nn as nn
import torch.nn.functional as F

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