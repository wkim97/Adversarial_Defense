###################################################################################################
# Implementation of Fast Smooth Patch Search Algorithm - noise extraction
# from Image Blind Denoising with GAN Based Noise Modeling - CVPR 2018
# Uses MNIST
###################################################################################################
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import cv2

gaussian_image_path = './noisy_images/gaussian'
pgd_image_path = './noisy_images/pgd_attacked'
gaussian_noise_path = './noise/gaussian'
pgd_noise_path = './noise/pgd_attacked'
use_gpu = torch.cuda.is_available()
batch_size = 1
mu = 0.1
gamma = 0.25

transforms = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor()])

gaussian_dataset = torchvision.datasets.ImageFolder(
    root=gaussian_image_path, transform=transforms)
gaussian_dataloader = torch.utils.data.DataLoader(
    gaussian_dataset, batch_size=batch_size, shuffle=True)

pgd_dataset = torchvision.datasets.ImageFolder(
    root=pgd_image_path, transform=transforms)
pgd_dataloader = torch.utils.data.DataLoader(
    pgd_dataset, batch_size=batch_size, shuffle=True)

# image is 1x28x28 tensor
# Returns list of patch_size x patch_size noise blocks
def extract_noise(image):
    temp_S = []
    image_size = image.size(2)
    image_mean = image.mean()
    image_var = image.std() ** 2
    patch_size = 14
    for i in range(int(image_size / patch_size)):
        for j in range(int(image_size / patch_size)):
            patch = image[0, 0, patch_size*i:patch_size*i+patch_size,
                    patch_size*j:patch_size*j+patch_size]
            if use_gpu:
                patch = patch.cuda()
            patch_mean = patch.mean()
            patch_var = patch.std() ** 2
            if (torch.abs(patch_mean - image_mean) <= mu * image_mean and
                torch.abs(patch_var - image_var) <= gamma * image_var):
                temp_S.append(patch)

    S = torch.FloatTensor(len(temp_S), patch_size, patch_size)
    V = torch.FloatTensor(len(temp_S), patch_size, patch_size)
    if use_gpu:
        S, V = S.cuda(), V.cuda()
    for i in range(len(temp_S)):
        S[i, :, :] = temp_S[i]

    for i in range(len(temp_S)):
        si_mean = temp_S[i].mean()
        mean = torch.zeros(patch_size, patch_size)
        mean = mean.fill_(si_mean)
        if use_gpu:
            mean, si_mean = mean.cuda(), si_mean.cuda()
        noise = temp_S[i] - mean
        noise[noise < 0.0] = 0.0
        V[i, :, :] = noise
    print(V)
    return V

# img = 0
# for image in gaussian_dataloader:
#     image = image[0]
#     noise = extract_noise(image)
#     num_patches = noise.size(0)
#     for i in range(num_patches):
#         save_dir = gaussian_noise_path + '/gaussian_{}.png'.format(img)
#         fig = plt.figure()
#         sample = np.transpose(vutils.make_grid(noise[i], normalize=True).cpu().detach().numpy(), (1, 2, 0))
#         plt.imsave(save_dir, sample, cmap="gray")
#         plt.close(fig)
#         img += 1

image = next(iter(gaussian_dataloader))[0]
img = 0
noise = extract_noise(image)
num_patches = noise.size(0)

sample = np.transpose(vutils.make_grid(image, normalize=True).cpu().detach().numpy(), (1, 2, 0))
plt.imsave(gaussian_noise_path + '/original.png', sample, cmap="gray")

image_size = image.size(2)
noise_image = torch.zeros(image_size, image_size)
for i in range(num_patches):
    save_dir = gaussian_noise_path + '/gaussian_{}.png'.format(img)
    fig = plt.figure()
    sample = np.transpose(vutils.make_grid(noise[i], normalize=True).cpu().detach().numpy(), (1, 2, 0))
    plt.imsave(save_dir, sample, cmap="gray")

    if i == 0:
        noise_image[0:14, 0:14] = noise[i]
    if i == 1:
        noise_image[0:14, 14:28] = noise[i]
    if i == 2:
        noise_image[14:28, 0:14] = noise[i]
    if i == 3:
        noise_image[14:28, 14:28] = noise[i]
    img += 1

fig = plt.figure()
sample = np.transpose(vutils.make_grid(noise_image, normalize=True).cpu().detach().numpy(), (1, 2, 0))
plt.imsave(gaussian_noise_path + '/noise.png', sample, cmap="gray")

model_noise = torch.zeros(1, 1 * 28 * 28)
model_noise.data.normal_(0, 1)
model_noise = model_noise.view(1, 28, 28)
sample = np.transpose(vutils.make_grid(model_noise, normalize=True).cpu().detach().numpy(), (1, 2, 0))
plt.imsave(gaussian_noise_path + '/model_noise.png', sample, cmap="gray")