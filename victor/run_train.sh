TOP=`pwd`/..
timestamp=`date +%Y%m%d%H%M%S`
cl=`git rev-parse HEAD|cut -c1-7`
name='L-DNQ'
cd $TOP
ex
train_cmd="python3 -m pdb train_base_model.py"

rm -fr Results
rm -fr *.log
python3 -m pdb MPSOGSA_ResNet.py --dataset $dataset --max_iters 30 --num_particles 20 --NumSave 6 2>&1 |tee Logs/${name}_train_${cl}_$timestamp.log
quantize_train="python3 -m pdb main.py"
#$quantize_train 2>&1 |tee Logs/${name}_quantize_${cl}_$timestamp.log
