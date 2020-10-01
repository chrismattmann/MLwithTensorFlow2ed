#!/usr/bin/env bash
# TODO: restrict gpu run to 127.0.0.1 again
if [[ "$(uname)" == "Linux" && -c /dev/nvidia0 ]]; then
    docker run --gpus all -p 127.0.0.1:8888:8888 --rm chrismattmann/mltf2:tf2
else
    docker run -p 127.0.0.1:8888:8888 --rm chrismattmann/mltf2:tf2
fi
