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
import os,sys
from ImageDataLoader import data_loading
DataPath='/git/data'
sys.path.append(DataPath)
print_device_useage=False
aresume=True
return_output=False

def ResumeModel(dataset,model_name,params,Monte_iter):
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/%s-%s-%s-%s-ckpt.pth' %(dataset,model_name,"para_{}_{}".format(params[0],params[1]),"Mon_{}".format(Monte_iter)))
    net = checkpoint['net']
    TrainConvergence = checkpoint['TrainConvergence']
    TestConvergence = checkpoint['TestConvergence']
    start_epoch = checkpoint['epoch']
    return net,TrainConvergence,TestConvergence,start_epoch

def print_nvidia_useage():
    global print_device_useage
    if print_device_useage:
        os.system('echo check gpu;nvidia-smi;echo check done')
    else:
        pass
    
# Training
def train(trainloader,net,optimizer,criterion,use_cuda):
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
    return TrainLoss,net

def test(testloader,net,criterion,use_cuda):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    TestLoss=[]
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
        test_loss_avg=test_loss/(batch_idx+1)
        print(batch_idx, len(testloader), 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss_avg, 100.*float(correct)/total, correct, total))
        TestLoss.append(test_loss_avg) 
    return TestLoss


def ResNet(dataset,params,Epochs,MentSize,lr,savepath):
    use_cuda = torch.cuda.is_available()
    best_loss = 100  # best test loss
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    model_name = 'ResNet20'
    
    TrainConvergence=[]
    TestConvergence=[]
    for Monte_iter in range(MentSize):
        # Data
        
        print_nvidia_useage()
        print('\n Monte calors times: %d' % Monte_iter)

        Batch_size=int(params[0])
        """trainloader = get_dataloader(dataset, 'train', Batch_size)
        testloader = get_dataloader(dataset, 'test', 100)"""
        
        trainloader = data_loading(DataPath,dataset,'train',Batch_size)
        testloader = data_loading(DataPath,dataset,'test', 100)
  
        # Model
        if dataset=='MNIST':
            if resume and os.path.exists('./checkpoint/%s-%s-%s-%s-ckpt.pth' %(dataset,model_name,"para_{}_{}".format(params[0],params[1]),"Mon_{}".format(Monte_iter))):
                [net,TrainConvergence,TestConvergence,start_epoch]=ResumeModel(dataset,model_name,params,Monte_iter)
            else:
                if params[1]==1:
                    net = Resnet20_MNIST()
                elif params[1]>0 and params[1]<1:
                    net = Resnet20_MNIST_Contraction(params[1]) 
                else:
                    print('Contraction coefficient should be [0,1], Now is: %d' % params[1])
                    pass
                
        elif dataset=='CIFAR10':
            if resume and os.path.exists('./checkpoint/%s-%s-%s-%s-ckpt.pth' %(dataset,model_name,"para_{}_{}".format(params[0],params[1]),"Mon_{}".format(Monte_iter))):
                [net,TrainConvergence,TestConvergence,start_epoch]=ResumeModel(dataset,model_name,params,Monte_iter)
            elif params[1]==1:
                net = resnet20_cifar()
            elif params[1]>0 and params[1]<1:
                net = resnet20_cifar_Contraction(params[1])

            else:
                print('Contraction coefficient should be [0,1], Now is: %d' % params[1])
                os.exit(0)

        if use_cuda:
            net.cuda()
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
 
        for epoch in range(start_epoch, start_epoch+Epochs):
            if epoch<Epochs:
                print('\nEpoch: %d' % epoch)
                [TrainLoss,net]=train(trainloader,net,optimizer,criterion,use_cuda)
                TrainConvergence.append(statistics.mean(TrainLoss))
                TestConvergence.append(statistics.mean(test(testloader,net,criterion,use_cuda)))
            else:
                break
                    # Save checkpoint.
            if TestConvergence[epoch] < best_loss:
                #print('Saving..')
                state = {
                    'net': net.module if use_cuda else net,
                    'TrainConvergence': TrainConvergence,
                    'TestConvergence': TestConvergence,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/%s-%s-%s-%s-ckpt.pth' %(dataset,model_name,"para_{}_{}".format(params[0],params[1]),"Mon_{}".format(Monte_iter)))

                best_loss = TestConvergence[epoch]
                if not os.path.exists('./%s' %model_name):
                    os.makedirs('./%s' %model_name)
                torch.save(net.module.state_dict(), './%s/%s_%s_pretrain.pth' %(model_name, dataset, model_name))
            else:
                pass
    
    
    FileName=dataset+str(params)+'TrainConvergenceChanges'+str(Monte_iter)
    np.save(savepath+FileName,TrainConvergence)
    
    FileName=dataset+str(params)+'TestConvergenceChanges'+str(Monte_iter)
    np.save(savepath+FileName,TestConvergence)

    if save_weight:
        return TestConvergence[-1], net.module.fc.weight
    else:
        pass

if __name__=="__main__":   
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--dataset',default='CIFAR10',type=str, help='dataset to train')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate') 
    parser.add_argument('--ConCoeff', default=1, type=float, help='contraction coefficients')
    parser.add_argument('--Epochs', default=1, type=int, help='Epochs')
    parser.add_argument('--MonteSize', default=1, type=int, help=' Monte Carlos size')
    parser.add_argument('--gpus', default="0", type=str, help="gpu devices")
    parser.add_argument('--BatchSize', default=512, type=int, help='Epochs')
    parser.add_argument('--savepath', type=str, default='Results/', help='Path to save results')
    parser.add_argument('--return_output', type=str, default=False, help='Whether output')
    parser.add_argument('--resume', '-r', type=str,default=False, help='resume from checkpoint')
    parser.add_argument('--print_device_useage', type=str, default=False, help='Whether print gpu useage')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print_device_useage=args.print_device_useage
    resume=args.resume
    return_output=args.return_output
    params=[args.BatchSize,args.ConCoeff]

    ResNet(args.dataset,params,args.Epochs,args.MonteSize,args.lr,args.savepath)

    

