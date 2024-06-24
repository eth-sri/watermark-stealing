#!/bin/bash

curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
rm Miniforge3-$(uname)-$(uname -m).sh
$HOME/miniforge3/bin/conda update -n base -c conda-forge conda

cd watermark-stealing
conda env create -f env.yaml
conda activate ws

# Install Flash attention (good luck)
pip install -U flash-attn --no-build-isolation

# Get PSP 
cd src/models/psp 
wget http://www.cs.cmu.edu/~jwieting/paraphrase-at-scale-english.zip .
unzip paraphrase-at-scale-english.zip
mv ./paraphrase-at-scale-english/* .

# Set huggingface token
huggingface-cli login