###################################################################################################
# Implementation of Fast Smooth Patch Search Algorithm - noise extraction
# from Image Blind Denoising with GAN Based Noise Modeling - CVPR 2018
# Uses MNIST
###################################################################################################
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import bm3d

gaussian_image_path = './noisy_images/gaussian/images'
pgd_image_path = './noisy_images/pgd_attacked/images'
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
    gaussian_dataset, batch_size=batch_size, shuffle=False)

pgd_dataset = torchvision.datasets.ImageFolder(
    root=pgd_image_path, transform=transforms)
pgd_dataloader = torch.utils.data.DataLoader(
    pgd_dataset, batch_size=batch_size, shuffle=False)


i = 0
for image in gaussian_dataloader:
    print("Creating Gaussian noise {}".format(i))
    image = image[0]
    if use_gpu:
        image = image.cuda()

    image = image.squeeze()
    denoised_image = bm3d.bm3d(image.cpu().numpy() + 0.5, sigma_psd=30 / 255,
                               stage_arg=bm3d.BM3DStages.ALL_STAGES)
    plt.imsave(gaussian_noise_path + '/denoised/images/denoised_{}.png'.format(i), denoised_image, cmap="gray")

    actual_noise = image.detach().cpu().numpy() - denoised_image
    plt.imsave(gaussian_noise_path + '/noise/images/noise_{}.png'.format(i), actual_noise, cmap="gray")
    i += 1

i = 0
for image in pgd_dataloader:
    print("Creating PGD noise {}".format(i))
    image = image[0]
    if use_gpu:
        image = image.cuda()

    image = image.squeeze()
    denoised_image = bm3d.bm3d(image.cpu().numpy() + 0.5, sigma_psd=30 / 255,
                               stage_arg=bm3d.BM3DStages.ALL_STAGES)
    plt.imsave(pgd_noise_path + '/denoised/images/denoised_{}.png'.format(i), denoised_image, cmap="gray")

    actual_noise = image.detach().cpu().numpy() - denoised_image
    plt.imsave(pgd_noise_path + '/noise/images/noise_{}.png'.format(i), actual_noise, cmap="gray")
    i += 1