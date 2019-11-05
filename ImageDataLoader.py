import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

""" usage DataPath='/git/data'
sys.path.append(DataPath)
from ImageDataLoader import data_loading
train_loader, test_loader = data_loading(DataPath,dataset,batch_size)
"""

def data_loading(DataPath,dataset,batch_size):
    
    if dataset == 'MNIST':
        train_dataset = getattr(dsets,dataset)(root=DataPath,  # 数据保持的位置
                                    train=True,  # 训练集
                                    transform=transforms.ToTensor(), 
                                    download=True)  # 下载数据,download首次为True

        test_dataset = getattr(dsets,dataset)(root=DataPath,
                                   train=False,  # 测试集
                                   transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        transform = transforms.Compose(
            [
             transforms.RandomHorizontalFlip(),
             transforms.RandomGrayscale(),
             transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])

        transform1 = transforms.Compose(
            [
             transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        
    if dataset == 'CIFAR10':
        train_dataset = getattr(dsets,dataset)(root=DataPath,  # 数据保持的位置
                                    train=True,  # 训练集
                                    transform=transforms.ToTensor(),  # 一个取值范围是[0,255]的PIL.Image
                                    # 转化为取值范围是[0,1.0]的torch.FloadTensor
                                    download=True)  # 下载数据,download首次为True

        test_dataset = getattr(dsets,dataset)(root=DataPath,
                                   train=False,  # 测试集
                                   transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        #  CIFAR10data数据集  下载训练集  CIFAR10data数据集训练集
        transform = transforms.Compose(
            [
             transforms.RandomHorizontalFlip(),
             transforms.RandomGrayscale(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        transform1 = transforms.Compose(
            [
             transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
    elif dataset == 'CIFAR100':
        train_dataset = getattr(dsets,dataset)(root=DataPath,  # 数据保持的位置
                                    train=True,  # 训练集
                                    transform=transforms.ToTensor(),  # 一个取值范围是[0,255]的PIL.Image
                                    # 转化为取值范围是[0,1.0]的torch.FloadTensor
                                    download=True)  # 下载数据,download首次为True

        test_dataset = getattr(dsets,dataset)(root=DataPath,
                                   train=False,  # 测试集
                                   transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        #  CIFAR10data数据集  下载训练集  CIFAR10data数据集训练集
        transform = transforms.Compose(
            [
             transforms.RandomHorizontalFlip(),
             transforms.RandomGrayscale(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        transform1 = transforms.Compose(
            [
             transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
        
    return train_loader,test_loader