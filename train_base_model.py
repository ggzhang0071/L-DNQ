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
import statistics
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
from models_CIFAR10.resnet import resnet20_cifar, resnet20_cifar_Contraction
from models_MNIST.resnet import Resnet20_MNIST, Resnet20_MNIST_Contraction
from torch.autograd import Variable
#from utils.train import progress_bar
from utils.dataset import get_dataloader
import SaveDataCsv as SV
import os,sys
from ImageDataLoader import data_loading
DataPath='/git/data'
sys.path.append(DataPath)


# Training
def train(trainloader,net,epoch,optimizer,criterion,use_cuda):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    TrainLoss=[]
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

        print(batch_idx, len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        TrainLoss.append(train_loss/(batch_idx+1)) 
    return TrainLoss

def test(dataset,testloader,net,epoch,criterion,best_acc,use_cuda,model_name):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    TestAcc=[]
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

        print(batch_idx, len(testloader), 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*float(correct)/total, correct, total))
        TestAcc.append(test_loss/(batch_idx+1)) 
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        #print('Saving..')
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
        torch.save(net.module.state_dict(), './%s/%s_%s_pretrain.pth' %(model_name, dataset, model_name))
    return TestAcc


def ResNet(dataset,params,Epochs,MentSize,lr,resume,savepath):
    use_cuda = torch.cuda.is_available()
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    model_name = 'ResNet20'
    
    TrainConvergence=np.zeros((MentSize,Epochs))
    TestConvergence=np.zeros((MentSize,Epochs))
    for k in range(MentSize):
        # Data
        print('==> Preparing data..')
        Batch_size=int(params[0])
        """trainloader = get_dataloader(dataset, 'train', Batch_size)
        testloader = get_dataloader(dataset, 'test', 100)"""
        
        trainloader = data_loading(DataPath,dataset,'train',Batch_size)
        testloader = data_loading(DataPath,dataset,'test', 100)
  
    
        # Model
        if dataset=='MNIST':
            if params[1]==1:
                net = Resnet20_MNIST()
            elif params[1]>0 and params[1]<1:
                net = Resnet20_MNIST_Contraction(params[1]) 
            else:
                #print('Contraction coefficient should be [0,1], Now is: %d',% params[1])
                break
                
        elif dataset=='CIFAR10':
            if params[1]==1:
                net = resnet20_cifar()
            elif params[1]>0 and params[1]<1:
                net = resnet20_cifar_Contraction(params[1])
                
            else:
                #print('Contraction coefficient should be [0,1], Now is: %d',% params[1])
                break
                
            """if resume:
                # Load checkpoint.
                print('==> Resuming from checkpoint..')
                assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
                checkpoint = torch.load('./checkpoint/%s_ckpt.t7' %model_name)
                net = checkpoint['net']
                best_acc = checkpoint['acc']
                start_epoch = checkpoint['epoch']"""


        if use_cuda:
            net.cuda()
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
 
        for epoch in range(start_epoch, start_epoch+Epochs):
            TrainConvergence[k,epoch]=statistics.mean(train(trainloader,net,epoch,optimizer,criterion,use_cuda))
            TestConvergence[k,epoch]=statistics.mean(test(dataset,testloader,net,epoch,criterion,best_acc,use_cuda,model_name))
    
    
    FileName=dataset+str(params)+'TrainConvergenceChanges'
    np.save(savepath+FileName,np.mean(TrainConvergence,0))
    
    FileName=dataset+str(params)+'TestConvergenceChanges'
    np.save(savepath+FileName,np.mean(TestConvergence,0))


    return TestConvergence[-1][-1], net.module.fc.weight

if __name__=="__main__":   
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset',default='CIFAR10',type=str, help='dataset to train')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate') 
    parser.add_argument('--ConCoeff', default=1, type=float, help='contraction coefficients')
    parser.add_argument('--Epochs', default=1, type=int, help='Epochs')
    parser.add_argument('--MentSize', default=1, type=int, help=' Monte Carlos size')
    parser.add_argument('--gpus', default="2,3", type=str, help="gpu devices")

    parser.add_argument('--BatchSize', default=512, type=int, help='Epochs')
    parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
    parser.add_argument('--savepath', type=str,required=False, default='Results/', help='Path to save results')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    params=[args.BatchSize,args.ConCoeff]

    [Acc,_]=ResNet(args.dataset,params,args.Epochs,args.MentSize,args.lr,args.resume,args.savepath)

    

