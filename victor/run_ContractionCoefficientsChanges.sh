TOP=`pwd`/..
timestamp=`date +%Y%m%d%H%M%S`
#cl=`git rev-parse HEAD|cut -c1-7`
name='L-DNQ_Contraction_Coefficients'
cd $TOP
#  'CIFAR10'
dataset='MNIST' 

#
for i in 0.1 0.2 0.4 0.6 
do
 echo "Contraction coefficients $i"
 python3 train_base_model.py --dataset $dataset --BatchSize 64 --ConCoeff $i --Epochs 80 --MentSize 5 2>&1 |tee Logs/${name}_${dataset}_$timestamp.log
done
 


name='L-DNQ_Batch_Size'
#
for i in 64 128 256 512 1024 
do
 echo "Batch size $i"
 python3  train_base_model.py --dataset $dataset --BatchSize $i --Epochs 80 --MentSize 20 2>&1 |tee Logs/${name}_${dataset}_$timestamp.log
done
