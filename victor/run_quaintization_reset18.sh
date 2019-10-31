TOP=`pwd`/..
timestamp=`date +%Y%m%d%H%M%S`
cl=`git rev-parse HEAD|cut -c1-7`
name='L-DNQ-ResNet18'
cd $TOP
mkdir ResNet18-ImageNet
wget https://download.pytorch.org/models/resnet18-5c106cde.pth
python3 main.py --model_name=ResNet18-ImageNet --dataset_name=CIFAR10
#python main.py --model_name=ResNet18-ImageNet 
