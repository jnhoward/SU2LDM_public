# SU2LDM_public

This repository contains the code used in the paper "Dark Matter Production from Weak Confinement": ...add arXiv link...


# Software Environment Setup

1. Make sure you have an anaconda3 installation with conda version 4.10.1 (or newer). To check this run the following on the command line and you should see similar outputs:
    
    $ which conda
    
    /Users/username/anaconda3/condabin/conda
    
    $ conda --version
    
    conda 4.10.1

2. Create a barebones Python 3.6.12 installation, including the (usually pre-installed) pip package manager which we'll use in step 3. There are many ways to do this (e.g., with virtualenv, or venv); we recommend the following setup with conda. Run the following on the command line:

    $ conda create --name py36-su2ldm python=3.6.12
    
    $ conda activate py36-su2ldm

3. After running conda activate py36-su2ldm, run the following to install the required packages.

    $ pip install -r coreRequirements.txt

    The main dependencies are listed in coreRequirements.txt. In order to avoid possible installation issues we have not listed their specific versions in coreRequirements.txt. However they are as follows:
    
    - ulysses==1.0.1
    - jupyter==1.0.0
    - numba==0.53.1
    - pandas==1.1.5
    
    If you would like to try to install these specific versions try the following
    
    $ pip install -r coreRequirements_specificVersions.txt
    
# Example of how to run ulysses

To run ulysses on a test data point, run the following on the command line:

$ uls-calc -m omegaH2_ulysses.py:SU2LDM test.dat
    
    