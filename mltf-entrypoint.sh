#!/usr/bin/env bash
set -e
# Install Py2 kernel
if [[ -f /usr/install/python27/bin/python ]]; then
    /usr/install/python27/bin/python -m ipykernel install --user
fi

# Start Jupyter and monitor it
jupyter notebook --notebook-dir=/usr/src/mltf2 --no-browser --ip  0.0.0.0 --allow-root

