TOP=`pwd`/..
timestamp=`date +%Y%m%d%H%M%S`
#cl=`git rev-parse HEAD|cut -c1-7`
name='L-DNQ'
cd $TOP
dataset='CIFAR10'

train_cmd="python3  MPSOGSA_ResNet.py"

rm -fr Results
rm -fr Logs

mkdir -p Results/RecurrencePlots Logs 
touch CIFAR10TestAccConvergenceChanges.csv Results/CIFAR10_BestParameters.csv
$train_cmd --dataset $dataset --max_iters 20 --num_particles 20 --Epochs 10 --NumSave 6 2>&1 |tee Logs/${name}_train_${cl}_$timestamp.log
=======
rm -fr *.log
#python3  MPSOGSA_ResNet.py --dataset $dataset --max_iters 30 --num_particles 20 --NumSave 6 2>&1 |tee Logs/${name}_train_${cl}_$timestamp.log
#python3   train_base_model.py --dataset $dataset  2>&1 |tee Logs/${name}_train_${cl}_$timestamp.log

quantize_train="python3 -m pdb main.py"
$quantize_train 2>&1 |tee Logs/${name}_quantize_${cl}_$timestamp.log
