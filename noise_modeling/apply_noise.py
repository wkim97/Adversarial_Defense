###################################################################################################
# Applies noise to MNIST dataset
###################################################################################################

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import advertorch.attacks as attacks
import matplotlib.pyplot as plt
import numpy as np
from adversarial_attacks.train_clean import MNIST_net

batch_size = 1
use_gpu = torch.cuda.is_available()
gaussian_path = './noisy_images/gaussian'
pgd_path = './noisy_images/pgd_attacked'
model_path = '../models/MNIST_net.pth'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])
trainset = torchvision.datasets.MNIST(
    '../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)
dataset = torchvision.datasets.MNIST(
    '../data', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True)

model = MNIST_net()
if use_gpu:
    model = model.cuda()
model.load_state_dict(torch.load(model_path))
linf_pgd_attack = attacks.LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                                        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
                                        clip_max=1.0, targeted=False)

def apply_gaussian(selected_class):
    i = 0
    for data in dataloader:
        images, labels = data
        if labels.numpy()[0] == selected_class:
            path = gaussian_path + '/{}/images/{}.png'.format(selected_class, i)
            b_size, ch, row, col = images.shape
            noise = torch.zeros(batch_size, ch * row * col)
            noise.data.normal_(0, 1)
            if use_gpu:
                images, labels, noise = images.cuda(), labels.cuda(), noise.cuda()
            images = images + noise.view(ch, row, col)
            fig = plt.figure()
            ax = plt.axis("off")
            sample = np.transpose(vutils.make_grid(images, normalize=True).cpu().detach().numpy(),
                                  (1, 2, 0))
            plt.imsave(path, sample, cmap="gray")
            plt.close(fig)
            i += batch_size
            print("Gaussian image {} for class {} created".format(i + 1, selected_class))
            if i == 5000:
                break

def apply_pgd(selected_class):
    i = 0
    for data in dataloader:
        images, labels = data
        if labels.numpy()[0] == selected_class:
            path = pgd_path + '/{}/images/{}.png'.format(selected_class, i)
            if use_gpu:
                images, labels = images.cuda(), labels.cuda()
            images = linf_pgd_attack.perturb(images, labels)
            fig = plt.figure()
            ax = plt.axis("off")
            sample = np.transpose(vutils.make_grid(images, normalize=True).cpu().detach().numpy(),
                                  (1, 2, 0))
            plt.imsave(path, sample, cmap="gray")
            plt.close(fig)
            i += batch_size
            print("PGD image {} for class {} created".format(i + 1, selected_class))
            if i == 5000:
                break

def main():
    for i in range(10):
        selected_class = i
        # apply_gaussian(selected_class)
        apply_pgd(selected_class)