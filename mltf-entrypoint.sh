#!/usr/bin/env bash

# Install Py2 kernel
/usr/install/python27/bin/python -m ipykernel install --user

# Start Jupyter and monitor it
jupyter notebook --notebook-dir=/usr/src/mltf2 --no-browser --ip  0.0.0.0 --allow-root

