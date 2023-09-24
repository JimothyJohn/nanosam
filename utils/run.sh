#!/usr/bin/env bash

docker run \
    -it \
    --rm \
    --ipc host \
    --gpus all \
    --shm-size 4G \
    -p 5000:5000 \
    -p 8888:8888 \
    -v $(pwd):/nanosam \
    -w /nanosam \
    nanosam:23-01

#    --device /dev/video0:/dev/video0 \
#    -v /tmp/.X11-unix:/tmp/.X11-unix \
#    -e DISPLAY=$DISPLAY \
