'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import pickle
import os

from models import *

from utils import progress_bar
import resnet20

from sat import *
import parameters


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if parameters.args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='~/datasets/pytorch-cifar', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='~/datasets/pytorch-cifar', train=False, download=True, transform=transform_test)
    num_classes = 10
else:
    trainset = torchvision.datasets.CIFAR100(root='~/datasets/pytorch-cifar', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='~/datasets/pytorch-cifar', train=False, download=True, transform=transform_test)
    num_classes = 100


trainloader = torch.utils.data.DataLoader(trainset, batch_size=parameters.args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dropout = True



# Model
print('==> Building model..')
net = resnet20.SATResNet20(num_classes = num_classes)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
print(net)
print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
if parameters.args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    parameters.temperature = parameters.temperature * (parameters.temperature_update**(start_epoch*391))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=parameters.args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch,optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs,nb_a = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 10 == 0 or batch_idx == 390:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | feature maps %d'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, nb_a))
        parameters.temperature = parameters.temperature * parameters.temperature_update
        
    
    

def test(epoch, write_file=False):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if batch_idx != 0:
                parameters.display = False
                parameters.heat_map = False
            # else:
            #     parameters.display = True
            inputs, targets = inputs.to(device), targets.to(device)
            outputs,ab_a = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 10 == 0 or batch_idx == 99:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if write_file:
        f = open("accuracy.csv", "a")
        if parameters.args.shared:
            shared = "shared"
        else:
            shared = "unshared"
        prints = [("dataset", parameters.args.dataset), ("initial temperature", str(parameters.args.temperature_init)), ("final temperature", str(parameters.args.temperature_final)), ("kernel size", str(parameters.args.kernel_size)), ("learning rate", str(parameters.args.lr)), ("epochs per era",str(parameters.args.epochs)), ("total parameters",str(parameters.nb_params)), ("shared shifts or not",shared), ("architecture",parameters.args.arch), ("dropout",str(parameters.args.dropout)), ("accuracy",str(acc))]
        f.write("#")
        for name,_ in prints:
            f.write(name + ", ")
        f.write("\n")
        for _,value in prints:
            f.write(value + ", ")
        f.write("\n")        
        f.close()
    if True:#>= 270 and not(parameters.args.resume):
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        
    if best_acc < acc:
        best_acc = acc
    print("best acc is : "+str(best_acc))

print("Total number of parameters after training will be: " + str(parameters.nb_params))
print("Temperature update is: " + str(parameters.temperature_update))
if not(parameters.args.resume):
    for epoch in range(start_epoch,parameters.args.epochs):
        if(epoch<100):
            optimizer = optim.SGD(net.parameters(), lr=parameters.args.lr, momentum=0.9, weight_decay=5e-4)
        elif(epoch<200):
            optimizer = optim.SGD(net.parameters(), lr=parameters.args.lr/10, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = optim.SGD(net.parameters(), lr=parameters.args.lr/100, momentum=0.9, weight_decay=5e-4)
        dropout = True
        train(epoch, optimizer)
        dropout = False
        test(epoch)
        print("Nb params: " + str(parameters.nb_params) + " and temperature is " + str(parameters.temperature) + " and max is " + str(parameters.maxvalue))

parameters.binarized = True
parameters.display = True

test(3*parameters.args.epochs,write_file=True)

