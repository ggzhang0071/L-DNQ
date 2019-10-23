TOP=`pwd`/..
timestamp=`date +%Y%m%d%H%M%S`
cl=`git rev-parse HEAD|cut -c1-7`
name='L-DNQ'
cd $TOP

train_cmd="python3 train_base_model.py"

rm -fr Results
rm -fr *.log
$train_cmd 2>&1 |tee ${name}_train_${cl}_$timestamp.log
quantize_train="python3 main.py"
$quantize_train 2>&1 |tee ${name}_quantize_${cl}_$timestamp.log

