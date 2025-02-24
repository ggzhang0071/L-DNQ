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
from pyts.image import RecurrencePlot
import numpy as np
import argparse
from models_CIFAR10.resnet import  Resnet20_CIFAR10
from models_MNIST.resnet import Resnet20_MNIST
from torch.autograd import Variable
#from utils.train import progress_bar
import os,sys
from ImageDataLoader import data_loading
DataPath='/git/data'
sys.path.append(DataPath)


def ResumeModel(model_to_save):
    # Load checkpoint.
    #print('==> Resuming from checkpoint..')
    checkpoint = torch.load(model_to_save)
    net = checkpoint['net']
    TrainConvergence = checkpoint['TrainConvergence']
    TestConvergence = checkpoint['TestConvergence']
    start_epoch = checkpoint['epoch']+1
    return net,TrainConvergence,TestConvergence,start_epoch

def logging(message):
    global  print_to_logging
    if print_to_logging:
        print(message)
    else:
        pass

    global print_device_useage
    if print_device_useage:
        os.system('echo check gpu;nvidia-smi;echo check done')
    else:
        pass

def print_nvidia_useage():
    global print_device_useage
    if print_device_useage:
        os.system('echo check gpu;nvidia-smi;echo check done')
    else:
        pass
    
def save_recurrencePlots(net,save_recurrencePlots_file):
    global save_recurrence_plots
    if save_recurrence_plots:
        for name,parameters in net.named_parameters():
            hiddenState=parameters.cpu().detach().numpy()
            if "fc" in name and hiddenState.ndim==2:
                rp = RecurrencePlot()
                X_rp = rp.fit_transform(hiddenState)
                plt.figure(figsize=(6, 6))
                plt.imshow(X_rp[0], cmap='binary', origin='lower')
                plt.savefig(save_recurrencePlots_file,dpi=600)
            else:
                continue
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
        predicted = torch.max(outputs.data, 1)[1]
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()
        TrainLoss.append(train_loss/(batch_idx+1))
        message="Batch number:{}, Train Loss: {} | Accuracy rate:{})".format(batch_idx, round(train_loss/(batch_idx+1),3), round(100.*correct/total,3))
        print(message) 

        del loss, predicted

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
        message="Batch number:{},Test Loss: {} | Accuracy rate:{}".format(batch_idx, round(test_loss_avg,3), round(100.*float(correct)/total,3))
        logging(message)
        TestLoss.append(test_loss_avg) 
    return TestLoss


def ResNet(dataset,params,Epochs,MonteSize,lr,savepath):
    
    use_cuda = torch.cuda.is_available()
    model_name = 'ResNet20'
  
    Batch_size=int(params[0])
    """trainloader = get_dataloader(dataset, 'train', Batch_size)
        testloader = get_dataloader(dataset, 'test', 100)"""
        

    for Monte_iter in range(MonteSize):
        trainloader = data_loading(DataPath,dataset,'train',Batch_size)
        testloader = data_loading(DataPath,dataset,'test', 100)
        # Data
        best_loss = float('inf')  # best test loss
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch         
        TrainConvergence=[]
        TestConvergence=[]

        # model 
        model_to_save='./checkpoint/{}-{}-param_{}_{}-Mon_{}-ckpt.pth'.format(dataset,model_name,params[0],params[1],Monte_iter)
        if dataset=='MNIST':
            if resume and os.path.exists(model_to_save):
                [net,TrainConvergence,TestConvergence,start_epoch]=ResumeModel(model_to_save)
                if start_epoch>=Epochs-1:
                    continue
                
            
            else:
                net=Resnet20_MNIST(params[1])
                
        elif dataset=='CIFAR10':
            if resume and os.path.exists(model_to_save):
                [net,TrainConvergence,TestConvergence,start_epoch]=ResumeModel(model_to_save)
                if start_epoch>=Epochs-1:
                    continue
            else: 
                net=Resnet20_CIFAR10(params[1])

        if use_cuda:
            net.cuda()
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = False

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        for epoch in range(start_epoch, start_epoch+Epochs):
            if epoch<Epochs:
                logging('Batch size: {},ConCoeff: {},MonteSize:{},epoch:{}'.format(params[0],params[1],Monte_iter,epoch))
                [TrainLoss,net]=train(trainloader,net,optimizer,criterion,use_cuda)
                TrainConvergence.append(statistics.mean(TrainLoss))
                TestConvergence.append(statistics.mean(test(testloader,net,criterion,use_cuda)))
            else:
                break
            if TestConvergence[epoch] < best_loss:
                logging('Saving..')
                state = {
                        'net': net.module if use_cuda else net,
                        'TrainConvergence': TrainConvergence,
                        'TestConvergence': TestConvergence,
                        'epoch': epoch,
                    }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, model_to_save)
                best_loss = TestConvergence[epoch]
                if not os.path.exists('./%s' %model_name):
                    os.makedirs('./%s' %model_name)
                torch.save(net.module.state_dict(), './%s/%s_%s_%s_pretrain.pth' %(model_name, dataset, model_name,params[1]))
            else:
                pass
            ## save recurrence plots
            if epoch%20==0 or epoch==Epochs-1:
                save_recurrencePlots_file="Results/RecurrencePlots/RecurrencePlots_{}_{}_BatchSize{}_ConCoeffi{}_epoch{}.png".format(dataset,
                                                                                                                                     model_name,params[0],params[1],epoch)
                                   
                save_recurrencePlots(net,save_recurrencePlots_file)
                                                                                                      
    
        FileName="{}-{}-param_{}_{}-monte_{}".format(dataset,model_name,params[0],params[1],Monte_iter)
        np.save(savepath+'TrainConvergence-'+FileName,TrainConvergence)
        np.save(savepath+'TestConvergence-'+FileName,TestConvergence)
        torch.cuda.empty_cache()
        print_nvidia_useage()


    if return_output==True:
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
    parser.add_argument('--resume', '-r', type=str,default=True, help='resume from checkpoint')
    parser.add_argument('--print_device_useage', type=str, default=False, help='Whether print gpu useage')
    parser.add_argument('--print_to_logging', type=str, default=True, help='Whether print')
    parser.add_argument('--save_recurrence_plots', type=str, default=False, help='Whether print')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print_to_logging=args.print_to_logging
    print_device_useage=args.print_device_useage
    resume=args.resume
    return_output=args.return_output
    save_recurrence_plots=args.save_recurrence_plots
    params=[args.BatchSize,args.ConCoeff]

    ResNet(args.dataset,params,args.Epochs,args.MonteSize,args.lr,args.savepath)

    

