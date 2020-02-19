import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import advertorch.attacks as attacks
import matplotlib.pyplot as plt
import numpy as np
from train_clean import MNIST_net
from tqdm import tqdm

path = os.getcwd()
print(path)

batch_size = 100
image_batch_size = 1
use_gpu = torch.cuda.is_available()

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
model_path = '../models/pgd_MNIST_net.pth'

model = MNIST_net()
if use_gpu:
    model = model.cuda()
model.load_state_dict(torch.load(model_path, map_location='cpu'))

###################################################################################################
# Set up Linf attacks
###################################################################################################
fgsm_attack = attacks.GradientSignAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                                         clip_min=0.0, clip_max=1.0, targeted=False)

bim_attack = attacks.LinfBasicIterativeAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                              eps=0.3, nb_iter=10, eps_iter=0.05, clip_min=0.0,
                                              clip_max=1.0, targeted=False)

linf_pgd_attack = attacks.LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                                        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
                                        clip_max=1.0, targeted=False)

###################################################################################################
# Set up L2 attacks
###################################################################################################
momentum_iterative_attack = attacks.MomentumIterativeAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                                            eps=0.3, nb_iter=40, decay_factor=1.0, eps_iter=0.01,
                                                            clip_min=0.0, clip_max=1.0, targeted=False)

cw_attack = attacks.CarliniWagnerL2Attack(model, num_classes=10, confidence=0, learning_rate=0.01,
                                          binary_search_steps=9, max_iterations=10000, abort_early=True,
                                          initial_const=0.001, clip_min=0.0, clip_max=1.0,
                                          loss_fn=nn.CrossEntropyLoss(reduction="sum"))

l2_pgd_attack = attacks.L2PGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                                    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
                                    clip_max=1.0, targeted=False)

###################################################################################################
# Set up L1 attacks
###################################################################################################
sparse_attack = attacks.SparseL1DescentAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                              eps=0.3, nb_iter=40, eps_iter=0.01, rand_init=False,
                                              clip_min=0.0, clip_max=1.0, l1_sparsity=0.95)

jsma_attack = attacks.JacobianSaliencyMapAttack(model, num_classes=10, clip_min=0.0, clip_max=1.0,
                                                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                                theta=1.0, gamma=1.0, comply_cleverhans=False)

elastic_net_attack = attacks.ElasticNetL1Attack(model, num_classes=10, loss_fn=nn.CrossEntropyLoss(reduction="sum"))

ddnl2_attack = attacks.DDNL2Attack(model, nb_iter=100, gamma=0.05, init_norm=1.0, quantize=True,
                                   levels=256, clip_min=0.0, clip_max=1.0, targeted=False,
                                   loss_fn=nn.CrossEntropyLoss(reduction="sum"))

lbfgs_attack = attacks.LBFGSAttack(model, num_classes=10, batch_size=1, binary_search_steps=9,
                                   max_iterations=100, initial_const=0.01, clip_min=0.0, clip_max=1.0,
                                   loss_fn=nn.CrossEntropyLoss(reduction="sum"), targeted=False)

single_pixel_attack = attacks.SinglePixelAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"))

local_search_attack = attacks.LocalSearchAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"))

spatial_transform_attack = attacks.SpatialTransformAttack(model, num_classes=10,
                                                          loss_fn=nn.CrossEntropyLoss(reduction="sum"))


attacks = {"clean" : None,
           "fgsm" : fgsm_attack,
           "bim" : bim_attack,
           "linf_pgd" : linf_pgd_attack,
           "momentum iterative" : momentum_iterative_attack,
           "cw" : cw_attack,
           "l2_pgd" : l2_pgd_attack,
           "sparse" : sparse_attack,
           "jsma" : jsma_attack,
           "elastic net" : elastic_net_attack,
           "ddnl2" : ddnl2_attack,
           "lbfgs" : lbfgs_attack,
           "single pixel" : single_pixel_attack,
           "spatial transform" : spatial_transform_attack}

def save_image(image, path):
    fig = plt.figure()
    sample = np.transpose(vutils.make_grid(image, normalize=True).cpu().detach().numpy(), (1, 2, 0))
    plt.imsave(path, sample, cmap="gray")
    plt.close(fig)


# attack = function
def test(attack, targeted):
    global attacks
    file_path = './results/adversarial_training_accuracy_results.csv'

    ###################################################################################################
    # Create file to store results
    ###################################################################################################
    f = open(file_path, 'a')

    ###################################################################################################
    # Classification results on clean dataset
    ###################################################################################################
    if attack == "clean":
        f.write("Clean dataset\n")
        correct = 0
        total = 0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with tqdm(total=(len(testset) - len(testset) % batch_size)) as _tqdm:
            _tqdm.set_description('{} attack: '.format(attack))
            for j, data in enumerate(testloader, 0):
                images, labels = data
                if use_gpu:
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                c = (predicted == labels).squeeze()
                for i in range(batch_size):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                avg_accuracy = 100 * correct / total
                _tqdm.set_postfix_str('Avg Accuracy: {:2.1f}'.format(avg_accuracy))
                _tqdm.update(batch_size)
        f.write('Avg Accuracy, %d %%\n'
                % (100 * correct / total))
        for i in range(10):
            f.write('Accuracy of %s, %2d %%\n'
                    % (classes[i], 100 * class_correct[i] / class_total[i]))


    ###################################################################################################
    # Classification results on attacked dataset
    ###################################################################################################
    else:
        if targeted:
            f.write("Targeted {} attack\n".format(attack))
        else:
            f.write("Untargeted {} attack\n".format(attack))
        correct = 0
        total = 0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        attack_func = attacks[attack]

        with tqdm(total=(len(testset) - len(testset) % batch_size)) as _tqdm:
            if targeted:
                _tqdm.set_description('Targeted {} attack: '.format(attack))
            else:
                _tqdm.set_description('Untargeted {} attack: '.format(attack))
            for j, data in enumerate(testloader, 0):
                images, labels = data
                if use_gpu:
                    images = images.cuda()
                    labels = labels.cuda()
                if targeted:
                    target = torch.ones_like(labels) * 3
                    attack_func.targeted = True
                    noisy_images = attack_func.perturb(images, target)
                else:
                    noisy_images = attack_func.perturb(images, labels)
                outputs = model(noisy_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                c = (predicted == labels).squeeze()
                for i in range(batch_size):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                avg_accuracy = 100 * correct / total
                _tqdm.set_postfix_str('Avg Accuracy: {}'.format(avg_accuracy))
                _tqdm.update(batch_size)
        f.write('Avg Accuracy, %d %%\n'
                % (100 * correct / total))
        for i in range(10):
            f.write('Accuracy of %s, %2d %%\n'
                    % (classes[i], 100 * class_correct[i] / class_total[i]))

    f.close()


for attack in attacks:
    if attack == "clean":
        test(attack, False)
    else:
        test(attack, False)
        test(attack, True)