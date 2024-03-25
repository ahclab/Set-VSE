#!/bin/bash

# docker image name
# DOCKER_IMAGE_NAME=$(basename "$(pwd)")"_base" # 1st parent dir name
#DOCKER_IMAGE_NAME=$(basename "$(dirname "$(pwd)")")"_base" # 2nd parent dir name
DOCKER_IMAGE_NAME="dci_base" # 2nd parent dir name

docker build \
        --no-cache=true \
        -t ${DOCKER_IMAGE_NAME} .
