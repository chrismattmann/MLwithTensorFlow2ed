#!/usr/bin/env bash

# Install Py2 kernel
/usr/install/python27/bin/python -m ipykernel install --user

# Start Jupyter and monitor it
jupyter notebook --no-browser --ip  0.0.0.0 

