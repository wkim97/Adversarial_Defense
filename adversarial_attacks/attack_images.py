import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import advertorch.attacks as attacks
import advertorch.test_utils as test_utils
from advertorch_examples.utils import TRAINED_MODEL_PATH

batch_size = 100

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])
batch_size = 100
trainset = torchvision.datasets.MNIST(
    './data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.MNIST(
    './data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

###################################################################################################
# Create training model
###################################################################################################
filename = "mnist_lenet5_clntrained.pt"

model = test_utils.LeNet5()
model.load_state_dict(
    torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))
model.cuda()
model.eval()
print(model)

###################################################################################################
# Set up PGD attack and create attacked images
###################################################################################################
pgd_attack = attacks.PGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                               nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
                               targeted=False)
clean_data, labels = next(iter(testloader))
adv_untargeted = adversary.perturb()