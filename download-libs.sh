#!/usr/bin/env bash

echo "Downloading Ch7 libraries..."
mkdir -p libs/BregmanToolkit/
curl -L "https://www.dropbox.com/sh/dkud00y61fq3ou7/AABYAwHiDmisx13rOAbvQzOWa?dl=0" -o libs/BregmanToolkit/BregmanToolkit.zip
pushd libs/BregmanToolkit/
unzip BregmanToolkit.zip
popd
echo "...done"

echo "Downloading Ch17 libraries..."
mkdir -p libs/basic_units/
curl -L "https://raw.githubusercontent.com/matplotlib/matplotlib/master/examples/units/basic_units.py" -o libs/basic_units/basic_units.py
pushd libs
curl -L "https://github.com/mrubash1/RNN-Tutorial/archive/master.zip" -o RNN-Tutorial.zip
unzip RNN-Tutorial.zip
mv RNN-Tutorial-master RNN-Tutorial && rm -rf RNN-Tutorial.zip
popd
echo "...done"

echo "Downloading Ch19 libraries..."
mkdir -p libs/vgg16
pushd libs/vgg16
curl -LO "https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz"
curl -LO "https://www.cs.toronto.edu/~frossard/vgg16/vgg16.py"
curl -LO "https://www.cs.toronto.edu/~frossard/vgg16/imagenet_classes.py"
curl -LO "https://www.cs.toronto.edu/~frossard/vgg16/laska.png"
popd
echo "...done"

