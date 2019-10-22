TOP=`pwd`/..
timestamp=`date +%Y%m%d%H%M%S`
cl=`git rev-parse HEAD|cut -c1-7`
name='L-DNQ'
cd $TOP

cmd="python3 train_base_model.py"

rm -fr Results
rm -fr *.log

$cmd 2>&1 |tee $name${cl}_$timestamp.log
$cmd 2>&1 |tee $name_${cl}_$timestamp.log

