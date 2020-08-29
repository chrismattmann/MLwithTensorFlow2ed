#!/usr/bin/env bash

if [[ "$(uname)" == "Linux" && -c /dev/nvidia0 ]]; then
    docker run --gpus all -p 127.0.0.1:8888:8888 --rm -it chrismattmann/mltf2:latest
else
    docker run -p 127.0.0.1:8888:8888 --rm -it chrismattmann/mltf2:latest
fi
