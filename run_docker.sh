#!/bin/bash

app=$PWD

docker build -t inception-v3 . && \
docker run -it --rm \
    --net=host --ipc=host --shm-size=2048m \
    --gpus "all" \
    -v "$app":/app \
    inception-v3
