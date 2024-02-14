# Convergence Rates Study with Reproducing Kernels

## Purpose

This repository contains code for conducting a convergence rates study with reproducing kernels. The purpose of this study is to analyze the convergence rates of algorithms utilizing reproducing kernels in a specific context. Users will find scripts and examples in the repository to understand and apply the findings.

## Installation

The code is written in Python 3.6. It is recommended setting up a virtual environment to manage dependencies. Follow these steps:

1. Create a virtual environment:

   ```bash
   python -m venv venv

On Windows:

    .\venv\Scripts\activate

On Unix or MacOS:
    
    source venv/bin/activate

Install dependencies: 
    
    pip install -r requirements.txt

 
Once the dependencies are installed, you can install the package using:
    
    cd <path to folder>
    pip install .

## Dependencies
The required packages are listed in the requirements.txt file. For more detailed information about each dependency, refer to the documentation:

1.  **Matplotlib:**
  - [Matplotlib Documentation](https://matplotlib.org/stable/users/index)

2.  **Numba:**
  - [Numba Documentation](https://numba.readthedocs.io/en/stable/index.html)

3.  **NumPy:**
  - [NumPy Documentation](https://numpy.org/doc/stable/)

4.  **SciPy:**
  - [SciPy Documentation](https://docs.scipy.org/doc/scipy/)

## License
This code is distributed under the ***Creative Commons*** license. See the LICENSE file for more details.

## Configuration
A configuration file inside the repository provides paths to the log and saving directory. Locate it by running:

    import rates.config
    print(rates.config.__file__)
Open the configuration file with your favorite text editor:

    vim <path to file>
Modify the configurations according to your needs.

## Usage
Examples and scripts are provided in the scripts folder, available on GitHub. Refer to these examples to understand how to use the code in your specific application.

## Quick Start
For a quick start, follow the steps in the "Installation" section and then refer to the provided examples in the scripts folder.

