TOP=`pwd`/..
timestamp=`date +%Y%m%d%H%M%S`
#cl=`git rev-parse HEAD|cut -c1-7`
cd $TOP
#  'CIFAR10'

rm -rf ../Logs/


dataset='MNIST' 

name='LDNQ_Batch_Size'
<<COMMENT
#
for i in 128 256 512 1024 
do
 echo "Batch size $i"
 python3  train_base_model.py --dataset $dataset --BatchSize $i --Epochs 80 --MonteSize 10 2>&1 |tee Logs/${name}_${dataset}_${i}_$timestamp.log
done
COMMENT
 
name='LDNQ_Contraction_Coefficients'


for i in  0.1 0.2
do
 echo "Contraction coefficients $i"
 python3  train_base_model.py --dataset $dataset --gpus '2,3' --BatchSize 512 --ConCoeff $i --Epochs 80 --MonteSize 10 --print_device_useage 'True' --resume 'True' --return_output 'False' 2>&1 |tee Logs/${name}_${dataset}_${i}_$timestamp.log
done



<<COMMENT

for i in 0.1 0.2 0.4 0.6 0.8
do
 echo "Contraction coefficients $i"
 python3   train_base_model.py --dataset $dataset --gpus '2,3' --BatchSize 512 --ConCoeff $i --Epochs 4 --MonteSize 2 --print_device_useage 'False' --resume 'True' --return_output 'False' 2>&1 |tee Logs/${name}_${dataset}_${i}_$timestamp.log
done
COMMENT

