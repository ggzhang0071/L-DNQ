import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

""" usage DataPath='/git/data'
sys.path.append(DataPath)
from ImageDataLoader import data_loading
train_loader, test_loader = data_loading(DataPath,dataset,batch_size)
"""

def data_loading(DataPath,dataset,split,batch_size):
    
    if dataset == 'MNIST':
        if split in ['train', 'limited']:
            transformTrain = transforms.Compose(
            [
             transforms.RandomHorizontalFlip(),
             transforms.RandomGrayscale(),
             transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])
            train_dataset = getattr(dsets,dataset)(root=DataPath,  # 数据保持的位置
                             train=True,  # 训练集
                             transform=transformTrain, 
                             download=True)  # 下载数据,download首次为True
                 
            Loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
            print ('Number of training data used: %d' %(len(Loader)))

            
        elif split == 'test' or split == 'val':
            transformTest = transforms.Compose(
              [
             transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

            test_dataset = getattr(dsets,dataset)(root=DataPath, train=False,  # 测试集
                                       transform=transformTest,
                                       download=True)

            Loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                       batch_size=batch_size,shuffle=False)
        
        
    if dataset == 'CIFAR10':
       
        #  CIFAR10data数据集  下载训练集  CIFAR10data数据集训练集
        if split in ['train', 'limited']:
            transformTrain = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.RandomGrayscale(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            train_dataset = getattr(dsets,dataset)(root=DataPath,  # 数据保持的位置
                                    train=True,  # 训练集
                                    transform=transformTrain,  # 一个取值范围是[0,255]的PIL.Image
                                    download=True)  # 下载数据,download首次为True
            Loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                   shuffle=True)

            print ('Number of training instances used: %d' %(len(Loader)))

            
        elif split == 'test' or split == 'val':
            transformTest = transforms.Compose(
            [
             transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            test_dataset = getattr(dsets,dataset)(root=DataPath,
                                   train=False,  # 测试集
                                   transform=transformTest,
                                   download=True)

            Loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
       

    elif dataset == 'CIFAR100':
        #  CIFAR10data数据集  下载训练集  CIFAR10data数据集训练集
        if split in ['train', 'limited']:
            transformTrain = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.RandomGrayscale(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            train_dataset = getattr(dsets,dataset)(root=DataPath,  # 数据保持的位置
                                    train=True,  # 训练集
                                    transform=transformTrain,  # 一个取值范围是[0,255]的PIL.Image
                                    download=True)  # 下载数据,download首次为True
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

            
        elif split == 'test' or split == 'val':
            transformTest = transforms.Compose(
            [
             transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            
        test_dataset = getattr(dsets,dataset)(root=DataPath,
                                   train=False,  # 测试集
                                   transform=transformTest,
                                     download=True)
       
        Loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        
    return Loader

if __name__=="__main__":
    Loader=data_loading('/git/data','CIFAR10','test',100)