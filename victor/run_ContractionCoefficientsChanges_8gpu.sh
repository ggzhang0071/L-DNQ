TOP=`pwd`/..
#timestamp=`date +%Y%m%d%H%M%S`
#cl=`git rev-parse HEAD|cut -c1-7`
name='L-DNQ_Contraction_Coefficients'
cd $TOP
#
dataset='CIFAR10' 
for i in  0.1  0.2 
do
 echo "Contraction coefficients $i"
 python3 train_base_model.py --gpus "0,1,2,3,4,5,6,7" --dataset $dataset --BatchSize 512 --ConCoeff $i --Epochs 80 --MonteSize 20  --print_device_useage 'True' --resume 'True' --return_output 'False' 2>&1 |tee Logs/${name}_${dataset}_${i}_$(date +%y%m%d%H%M%S).log
done
 
#  'CIFAR10'
dataset='MNIST' 
#
for i in 0.1 0.2 0.4 0.6 0.8
do
 echo "Contraction coefficients $i"
 python3 train_base_model.py --gpus "0,1,2,3,4,5,6,7" --dataset $dataset --BatchSize 512 --ConCoeff $i --Epochs 80 --MonteSize 20  --print_device_useage 'True' --resume 'True' --return_output 'False' 2>&1 |tee Logs/${name}_${dataset}_${i}_$(date +%y%m%d%H%M%S).log
done
 


name='L-DNQ_Batch_Size'
#
for i in 64 128 256 512 1024 
do
 echo "Batch size $i"
 python3 train_base_model.py --gpus "0,1,2,3,4,5,6,7" --dataset $dataset --BatchSize 512 --ConCoeff $i --Epochs 80 --MonteSize 20  --print_device_useage 'True' --resume 'True' --return_output 'False' 2>&1 |tee Logs/${name}_${dataset}_${i}_$(date +%y%m%d%H%M%S).log
done
