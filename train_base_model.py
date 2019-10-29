#from __future__ import print_function
"""Train CIFAR10 with PyTorch."""
"""
This code is forked and modified from 'https://github.com/kuangliu/pytorch-cifar'. Thanks to its contribution.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import argparse

from models_CIFAR10.resnet import resnet20_cifar
from torch.autograd import Variable
from utils.train import progress_bar
from utils.dataset import get_dataloader


# Training
def train(trainloader,net,epoch,optimizer,criterion,use_cuda):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        progress_bar(batch_idx, len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(testloader,net,epoch,criterion,best_acc,use_cuda,model_name):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    Acc=[]
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        progress_bar(batch_idx, len(testloader), 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*float(correct)/total, correct, total))
        Acc.append(100.*float(correct)/total) 
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/%s_ckpt.t7' %model_name)
        best_acc = acc
        if not os.path.exists('./%s' %model_name):
            os.makedirs('./%s' %model_name)
        torch.save(net.module.state_dict(), './%s/%s_pretrain.pth' %(model_name, model_name))
    return Acc[-1]

def ResNet(dataset,params,lr,resume,savepath):
    use_cuda = torch.cuda.is_available()
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    model_name = 'ResNet20'

    # Data
    print('==> Preparing data..')
    Batch_size=int(params[0])
    trainloader = get_dataloader(dataset, 'train', Batch_size)
    testloader = get_dataloader(dataset, 'test', 100)

    # Model
    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/%s_ckpt.t7' %model_name)
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        print('==> Building model..')
        # net = VGG(model_name)
        # net = ResNet18()
        # net = PreActResNet18()
        # net = GoogLeNet()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        # net = MobileNetV2()
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()
        # net = Wide_ResNet(**{'widen_factor':20, 'depth':28, 'dropout_rate':0.3, 'num_classes':10})
        # net = resnet32_cifar()
        # net = resnet56_cifar()
        net = resnet20_cifar(params[1])

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for epoch in range(start_epoch, start_epoch+200):
        train(trainloader,net,epoch,optimizer,criterion,use_cuda)
        Acc=test(testloader,net,epoch,criterion,best_acc,use_cuda,model_name)
    return Acc, net.module.fc.weight

if __name__=="__main__":   
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset',default='CIFAR10',type=str, help='dataset to train')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
    parser.add_argument('--savepath', type=str,required=False, default='./Results/',
                    help='Path to save results')
    args = parser.parse_args()
    params=[np.random.randint(300*0.4,300),np.random.uniform(0.5,0.9)]

    ResNet(args.dataset,params,args.lr,args.resume,args.savepath)