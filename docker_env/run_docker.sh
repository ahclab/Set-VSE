#!/bin/bash

ROOT_DIR=$(dirname "$(pwd)") # set grandparent directory as rootdir
CONTAINER_HOME=/workspace/myspace

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi

PROJECT_NAME="dci"
DOCKER_IMAGE="${PROJECT_NAME}_base"
DATE=$(date "+%Y-%m-%d-%H-%M-%S")
CONTAINER_NAME="${PROJECT_NAME}_${DATE}"

echo "Launching a container named ${CONTAINER_NAME}"
docker run -it --rm \
    --gpus "device=$CUDA_VISIBLE_DEVICES" \
    --name  ${CONTAINER_NAME} \
    --mount src=${ROOT_DIR},dst=${CONTAINER_HOME},type=bind \
    --shm-size=124G \
    -p 8888:8888 \
    --env WORKDIR=${CONTAINER_HOME}  \
    ${DOCKER_IMAGE}:latest \
    /bin/bash 


