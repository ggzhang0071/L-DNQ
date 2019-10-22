img="nvcr.io/nvidia/tensorflow:19.09-py3"
TOP=`pwd`/..

nvidia-docker run --privileged=true  -e DISPLAY  --net=host --ipc=host -d --rm  -p 7022:22 -p 5022:5022 \
     -v $TOP:/wrk \
     -w /wrk  \
     $img  tail -f /dev/null


