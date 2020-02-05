import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import advertorch.attacks as attacks
from adversarial_attacks.train_clean import MNIST_net
from adversarial_attacks.train_pgd import pgd_MNIST_net

batch_size = 100
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
file_path = ''
model_path = ''
use_clean_model = False
model = MNIST_net()

if use_clean_model:
    model = MNIST_net()
    model_path = './models/MNIST_net.pth'
    file_path = './results/attack_on_clean_model_accuracy_results.csv'
else:
    model = pgd_MNIST_net()
    model_path = './models/pgd_MNIST_net.pth'
    file_path = './results/attack_on_pgd_model_accuracy_results.csv'
if use_gpu:
    model = model.cuda()
model.load_state_dict(torch.load(model_path))

###################################################################################################
# Set up Linf attacks
###################################################################################################
fgsm_attack = attacks.GradientSignAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.1,
                                         clip_min=0.0, clip_max=1.0, targeted=False)

bim_attack = attacks.LinfBasicIterativeAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                              eps=0.1, nb_iter=10, eps_iter=0.05, clip_min=0.0,
                                              clip_max=1.0, targeted=False)

linf_pgd_attack = attacks.LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                                        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
                                        clip_max=1.0, targeted=False)

###################################################################################################
# Set up L2 attacks
###################################################################################################
cw_attack = attacks.CarliniWagnerL2Attack(model, num_classes=10, confidence=0, targeted=False,
                                          learning_rate=0.01, binary_search_steps=9,
                                          max_iterations=10000, abort_early=True, initial_const=0.001,
                                          clip_min=0.0, clip_max=1.0)

l2_pgd_attack = attacks.L2PGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                                    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
                                    clip_max=1.0, targeted=False)

###################################################################################################
# Set up L1 attacks
###################################################################################################
jsma_attack = attacks.JacobianSaliencyMapAttack(model, num_classes=10, clip_min=0.0, clip_max=1.0,
                                                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                                theta=1.0, gamma=0.1, comply_cleverhans=False)

l0_pgd_attack = attacks.L1PGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                                    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
                                    clip_max=1.0, targeted=False)

ddnl2_attack = attacks.DDNL2Attack(model, nb_iter=100, gamma=0.05, init_norm=1.0, quantize=True,
                                   levels=256, clip_min=0.0, clip_max=1.0, targeted=False,
                                   loss_fn=nn.CrossEntropyLoss(reduction="sum"))

lbfgs_attack = attacks.LBFGSAttack(model, num_classes=10, batch_size=1, binary_search_steps=9,
                                   max_iterations=100, initial_const=0.01, clip_min=0.0, clip_max=1.0,
                                   loss_fn=nn.CrossEntropyLoss(reduction="sum"), targeted=False)

###################################################################################################
# Create file to store results
###################################################################################################
f = open(file_path, 'w')

###################################################################################################
# Classification results on clean dataset
###################################################################################################
f.write("Clean dataset\n")
print("Clean dataset...")
correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
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
f.write('Avg Accuracy, %d %%\n'
        % (100 * correct / total))
for i in range(10):
    f.write('Accuracy of %s, %2d %%\n'
            % (classes[i], 100 * class_correct[i] / class_total[i]))

###################################################################################################
# Classification results on FGSM attacked dataset
###################################################################################################
f.write("L_inf attack methods\n")
f.write("FGSM attack\n")
print("FGSM attack...")
correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for j, data in enumerate(testloader, 0):
    images, labels = data
    if use_gpu:
        images = images.cuda()
        labels = labels.cuda()
    fgsm_images = fgsm_attack.perturb(images, labels)
    outputs = model(fgsm_images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    c = (predicted == labels).squeeze()
    for i in range(batch_size):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1
f.write('Avg Accuracy, %d %%\n'
        % (100 * correct / total))
for i in range(10):
    f.write('Accuracy of %s, %2d %%\n'
            % (classes[i], 100 * class_correct[i] / class_total[i]))

###################################################################################################
# Classification results on BIM attacked dataset
###################################################################################################
f.write("BIM attack\n")
print("BIM attack...")
correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for j, data in enumerate(testloader, 0):
    images, labels = data
    if use_gpu:
        images = images.cuda()
        labels = labels.cuda()
    bim_images = bim_attack.perturb(images, labels)
    outputs = model(bim_images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    c = (predicted == labels).squeeze()
    for i in range(batch_size):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1
f.write('Avg Accuracy, %d %%\n'
        % (100 * correct / total))
for i in range(10):
    f.write('Accuracy of %s, %2d %%\n'
            % (classes[i], 100 * class_correct[i] / class_total[i]))

###################################################################################################
# Classification results on untargeted PGD attacked dataset
###################################################################################################
f.write("Untargeted Linf_PGD attack\n")
print("Untargeted Linf PGD attack...")
correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for j, data in enumerate(testloader, 0):
    images, labels = data
    if use_gpu:
        images = images.cuda()
        labels = labels.cuda()
    untargeted_pgd_images = linf_pgd_attack.perturb(images, labels)
    outputs = model(untargeted_pgd_images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    c = (predicted == labels).squeeze()
    for i in range(batch_size):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1
f.write('Avg Accuracy, %d %%\n'
        % (100 * correct / total))
for i in range(10):
    f.write('Accuracy of %s, %2d %%\n'
            % (classes[i], 100 * class_correct[i] / class_total[i]))

###################################################################################################
# Classification results on targeted PGD attacked dataset
###################################################################################################
f.write("Targeted Linf_PGD attack\n")
print("Targeted Linf PGD attack...")
correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for j, data in enumerate(testloader, 0):
    images, labels = data
    if use_gpu:
        images = images.cuda()
        labels = labels.cuda()
    target = torch.ones_like(labels) * 3
    linf_pgd_attack.targeted = True
    targeted_pgd_images = linf_pgd_attack.perturb(images, target)
    outputs = model(targeted_pgd_images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    c = (predicted == labels).squeeze()
    for i in range(batch_size):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1
f.write('Avg Accuracy, %d %%\n'
        % (100 * correct / total))
for i in range(10):
    f.write('Accuracy of %s, %2d %%\n'
            % (classes[i], 100 * class_correct[i] / class_total[i]))

###################################################################################################
# Classification results on untargeted CW attacked dataset
###################################################################################################
# f.write("L2 attack methods\n")
# f.write("Untargeted CW attack\n")
# print("Untargeted CW attack...")
# correct = 0
# total = 0
# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# for j, data in enumerate(testloader, 0):
#     if j == 5:
#         print(j)
#         break
#     images, labels = data
#     if use_gpu:
#         images = images.cuda()
#         labels = labels.cuda()
#     untargeted_cw_images = cw_attack.perturb(images, labels)
#     outputs = model(untargeted_cw_images)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum().item()
#     c = (predicted == labels).squeeze()
#     for i in range(batch_size):
#         label = labels[i]
#         class_correct[label] += c[i].item()
#         class_total[label] += 1
# f.write('Avg Accuracy, %d %%\n'
#         % (100 * correct / total))
# for i in range(10):
#     f.write('Accuracy of %s, %2d %%\n'
#             % (classes[i], 100 * class_correct[i] / class_total[i]))

###################################################################################################
# Classification results on targeted CW attacked dataset
###################################################################################################
# f.write("Targeted CW attack\n")
# print("Targeted CW attack...")
# correct = 0
# total = 0
# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# for j, data in enumerate(testloader, 0):
#     if j == 5:
#         print(j)
#         break
#     images, labels = data
#     if use_gpu:
#         images = images.cuda()
#         labels = labels.cuda()
#     target = torch.ones_like(labels) * 3
#     cw_attack.targeted = True
#     targeted_cw_images = cw_attack.perturb(images, target)
#     outputs = model(targeted_cw_images)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum().item()
#     c = (predicted == labels).squeeze()
#     for i in range(batch_size):
#         label = labels[i]
#         class_correct[label] += c[i].item()
#         class_total[label] += 1
# f.write('Avg Accuracy, %d %%\n'
#         % (100 * correct / total))
# for i in range(10):
#     f.write('Accuracy of %s, %2d %%\n'
#             % (classes[i], 100 * class_correct[i] / class_total[i]))

###################################################################################################
# Classification results on untargeted L2_PGD attacked dataset
###################################################################################################
f.write("Untargeted L2_PGD attack\n")
print("Untargeted L2 PGD attack...")
correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for j, data in enumerate(testloader, 0):
    images, labels = data
    if use_gpu:
        images = images.cuda()
        labels = labels.cuda()
    untargeted_L2pgd_images = l2_pgd_attack.perturb(images, labels)
    outputs = model(untargeted_L2pgd_images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    c = (predicted == labels).squeeze()
    for i in range(batch_size):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1
f.write('Avg Accuracy, %d %%\n'
        % (100 * correct / total))
for i in range(10):
    f.write('Accuracy of %s, %2d %%\n'
            % (classes[i], 100 * class_correct[i] / class_total[i]))

###################################################################################################
# Classification results on targeted L2_PGD attacked dataset
###################################################################################################
f.write("Targeted L2_PGD attack\n")
print("Targeted L2 PGD attack...")
correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for j, data in enumerate(testloader, 0):
    images, labels = data
    if use_gpu:
        images = images.cuda()
        labels = labels.cuda()
    target = torch.ones_like(labels) * 3
    l2_pgd_attack.targeted = True
    targeted_l2pgd_images = l2_pgd_attack.perturb(images, target)
    outputs = model(targeted_l2pgd_images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    c = (predicted == labels).squeeze()
    for i in range(batch_size):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1
f.write('Avg Accuracy, %d %%\n'
        % (100 * correct / total))
for i in range(10):
    f.write('Accuracy of %s, %2d %%\n'
            % (classes[i], 100 * class_correct[i] / class_total[i]))

###################################################################################################
# Classification results on untargeted JSMA attacked dataset
###################################################################################################
f.write("L0 attack methods\n")
f.write("Untargeted JSMA attack\n")
print("Untargeted JSMA attack...")
correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for j, data in enumerate(testloader, 0):
    images, labels = data
    if use_gpu:
        images = images.cuda()
        labels = labels.cuda()
    untargeted_jsma_images = jsma_attack.perturb(images, labels)
    outputs = model(untargeted_jsma_images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    c = (predicted == labels).squeeze()
    for i in range(batch_size):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1
f.write('Avg Accuracy, %d %%\n'
        % (100 * correct / total))
for i in range(10):
    f.write('Accuracy of %s, %2d %%\n'
            % (classes[i], 100 * class_correct[i] / class_total[i]))

###################################################################################################
# Classification results on targeted JSMA attacked dataset
###################################################################################################
f.write("Targeted JSMA attack\n")
print("Targeted JSMA attack...")
correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for j, data in enumerate(testloader, 0):
    images, labels = data
    if use_gpu:
        images = images.cuda()
        labels = labels.cuda()
    target = torch.ones_like(labels) * 3
    jsma_attack.targeted = True
    targeted_jsma_images = jsma_attack.perturb(images, target)
    outputs = model(targeted_jsma_images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    c = (predicted == labels).squeeze()
    for i in range(batch_size):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1
f.write('Avg Accuracy, %d %%\n'
        % (100 * correct / total))
for i in range(10):
    f.write('Accuracy of %s, %2d %%\n'
            % (classes[i], 100 * class_correct[i] / class_total[i]))

###################################################################################################
# Classification results on DDNL2 attacked dataset
###################################################################################################
f.write("L0 attack methods\n")
f.write("Decoupled Direction and Norm attack\n")
print("Decoupled Direction and Norm attack...")
correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for j, data in enumerate(testloader, 0):
    images, labels = data
    if use_gpu:
        images = images.cuda()
        labels = labels.cuda()
    ddnl2_images = ddnl2_attack.perturb(images, labels)
    outputs = model(ddnl2_images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    c = (predicted == labels).squeeze()
    for i in range(batch_size):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1
f.write('Avg Accuracy, %d %%\n'
        % (100 * correct / total))
for i in range(10):
    f.write('Accuracy of %s, %2d %%\n'
            % (classes[i], 100 * class_correct[i] / class_total[i]))

###################################################################################################
# Classification results on LBFGS attacked dataset
###################################################################################################
f.write("L-BFGS attack\n")
print("L-BFGS attack...")
correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for j, data in enumerate(testloader, 0):
    images, labels = data
    if use_gpu:
        images = images.cuda()
        labels = labels.cuda()
    lbfgs_images = lbfgs_attack.perturb(images, labels)
    outputs = model(lbfgs_images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    c = (predicted == labels).squeeze()
    for i in range(batch_size):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1
f.write('Avg Accuracy, %d %%\n'
        % (100 * correct / total))
for i in range(10):
    f.write('Accuracy of %s, %2d %%\n'
            % (classes[i], 100 * class_correct[i] / class_total[i]))

print("Testing complete!!!")
f.close()