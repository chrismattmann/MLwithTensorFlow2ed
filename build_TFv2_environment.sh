#!/usr/bin/env bash
# rm -f requirements.txt
# pipenv run pip freeze | grep -v horovod > requirements.txt
docker build --rm --pull --tag chrismattmann/mltf2:tf2 -f Dockerfile-TFv2 .

