###################################################################################################
# Implementation of Fast Smooth Patch Search Algorithm - noise extraction
# from Image Blind Denoising with GAN Based Noise Modeling - CVPR 2018
# Uses MNIST
###################################################################################################
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import advertorch.attacks as attacks
import matplotlib as plt
import numpy as np
from adversarial_attacks.train_clean import MNIST_net

gaussian_path = './noisy_images/gaussian'
pgd_path = './noisy_images/pgd_attacked'
use_gpu = torch.cuda.is_available()
batch_size = 50

transforms = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor()])

gaussian_dataset = torchvision.datasets.ImageFolder(
    root=gaussian_path, transform=transforms)
gaussian_dataloader = torch.utils.data.DataLoader(
    gaussian_dataset, batch_size=batch_size, shuffle=True)

pgd_dataset = torchvision.datasets.ImageFolder(
    root=pgd_path, transform=transforms)
pgd_dataloader = torch.utils.data.DataLoader(
    pgd_dataset, batch_size=batch_size, shuffle=True)

