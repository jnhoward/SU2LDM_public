# SU2LDM_public

This repository contains the code used in the paper "Title": <add arXiv link>.

# Software Setup

## Software Dependencies

## Software Environment
To reproduce the Python environment, do the following:

1. Create a barebones Python 3.6.9 installation, including the (usually pre-installed) pip package manager which we'll use in step 2. There are many ways to do this (e.g., with virtualenv, or venv); we recommend the following setup with conda: conda create --name py36-su2ldm python=3.6.9; conda activate py36-su2ldm. conda will make sure to set up pip within this new environment.

2. After running conda activate py36-su2ldm, run pip install -r requirements.txt to install the required packages.

The main dependencies are Python 3.6.9, Numpy 1.17.4, and Jupyter 1.0 (see requirements-core.txt). You may use other versions of these packages, but may not be able to exactly reproduce the reported results.