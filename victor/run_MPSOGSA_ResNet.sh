timestamp=`date +%Y%m%d%H%M%S`

dataset='CIFAR10'

python -m pdb  ../MPSOGSA_ResNet.py  --dataset $dataset  --max_iters 30 --num_particles 20  --NumSave 6   |tee ../Logs/PSOGSA_ResNet$dataset_$timestamp.log