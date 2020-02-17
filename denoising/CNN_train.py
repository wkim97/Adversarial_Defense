import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from denoising.utils import AverageMeter
from denoising.CNN_model import MNIST_net

batch_size = 100
use_gpu = torch.cuda.is_available()
num_epochs = 100

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])
trainset = torchvision.datasets.MNIST(
    '../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.MNIST(
    '../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
model_path = './models/clean_Cnn_model.pth'

###################################################################################################
# Create training model
###################################################################################################
model = MNIST_net()
if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

###################################################################################################
# Train the model
###################################################################################################
def train():
    for epoch in range(num_epochs):
        epoch_losses = AverageMeter()

        with tqdm(total=(len(trainset) - len(trainset) % batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, num_epochs))
            for i, data in enumerate(trainloader, 0):
                images, labels = data
                if use_gpu:
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = model(images)

                loss = criterion(outputs, labels)

                epoch_losses.update(loss.item(), len(images))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(images))
    print('Finished Training')
    torch.save(model.state_dict(), model_path)

train()