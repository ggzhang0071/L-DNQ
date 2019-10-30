TOP=`pwd`/..
timestamp=`date +%Y%m%d%H%M%S`
#cl=`git rev-parse HEAD|cut -c1-7`
name='L-DNQ_contraction_coefficients'
cd $TOP
dataset='CIFAR10'

for i in 0.4  0.6 0.8 
do
  python3 train_base_model.py --ConCoeff $i   2>&1 |tee Logs/${name}_train_${dataset}_$timestamp.log
done