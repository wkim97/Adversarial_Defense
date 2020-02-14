import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from random import randint

use_gpu = torch.cuda.is_available()
batch_size = 1
store_path = './noisy_images'
read_path = './noise_models/pgd_attacked/noise_models'

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])
dataset = torchvision.datasets.MNIST(
    '../data', train=True, download=True, transform=transforms)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True)

def apply_noise():
    for data in dataloader:
        image, label = data
        if use_gpu:
            image, label = image.cuda(), label.cuda()
        selected_label = label.detach().cpu().numpy()[0]
        for i in range(10):
            index = random()




apply_noise()