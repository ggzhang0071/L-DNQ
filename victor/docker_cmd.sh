img="nvcr.io/nvidia/tensorflow:19.01-py3"
TOP=`pwd`/..

nvidia-docker run --privileged=true  -e DISPLAY  --net=host --ipc=host -d --rm  -p 7022:22 -p 5022:5022 \
      -v $TOP:/git \
      -p 6033:8888 \
     $img  tail -f /dev/null


