# Convergence rates study with kernels

This repository contains the code for the convergence rates study with reproducing kernels. 

## Installation
The code is written in Python 3.6. The required packages are listed in `requirements.txt`. To install them, run:
```
pip install -r requirements.txt
```
Once the dependencies are installed, the package can be installed with
```
cd <path to folder>
pip install .
```

#### Configuration
A configuration file inside the repository provide path to log and saving directory.
It can be located by launching a python instance with the command line `python`, and executing:
```
import rates.config
print(rates.config.__file__)
```
This configuration can be open with your favorite text editor and modified to your needs.
```
vim <path to file>
```

## Usage
Examples are provided in the `scripts` folder, available on GitHub.
