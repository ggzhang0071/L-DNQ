img="nvcr.io/nvidia/tensorflow:19.01-py3"
TOP=`pwd`/..

nvidia-docker run --privileged=true  -e DISPLAY --ipc=host -d --rm  -p 6043:8888 \
      -v $TOP:/git \
     $img  tail -f /dev/null


