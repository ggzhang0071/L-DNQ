TOP=`pwd`/..
timestamp=`date +%Y%m%d%H%M%S`
#cl=`git rev-parse HEAD|cut -c1-7`
name='L-DNQ_contraction_coefficients'
cd $TOP
dataset='CIFAR10'

for i in 0.1 0.2 0.4 0.6 
do
 echo "Contraction coefficients $i"
 python3 -m pdb  train_base_model.py --ConCoeff $i --Epochs 10 --MentSize 10 2>&1 |tee Logs/${name}_${dataset}_$timestamp.log
done

<<'COMMENT'
for i in 64 128 256 512 1024 
do
 echo "Batch size $i"
 python3  train_base_model.py --BatchSize $i --Epochs 80 --MentSize 10 2>&1 |tee Logs/${name}_${dataset}_$timestamp.log
done
COMMENT