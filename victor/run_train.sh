TOP=`pwd`/..
timestamp=`date +%Y%m%d%H%M%S`
#cl=`git rev-parse HEAD|cut -c1-7`
name='L-DNQ'
cd $TOP
dataset='CIFAR10'

train_cmd="python3  MPSOGSA_ResNet.py"


mkdir -p Results/RecurrencePlots Logs 
$train_cmd --dataset $dataset --max_iters 5 --num_particles 5 --gpus '2,3' --Epochs 80 --NumSave 6 --savepath 'Results/' 2>&1 |tee Logs/${name}_train_${cl}_$timestamp.log

#python3   train_base_model.py --dataset $dataset  2>&1 |tee Logs/${name}_train_${cl}_$timestamp.log

#quantize_train="python3 -m pdb main.py"
#$quantize_train --batch_size 119 2>&1 |tee Logs/${name}_quantize_${cl}_$timestamp.log
