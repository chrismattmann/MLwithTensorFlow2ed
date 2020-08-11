#!/usr/bin/env bash
# rm -f requirements.txt
# pipenv run pip freeze | grep -v horovod > requirements.txt
docker build --rm --pull --tag mltf2 .
