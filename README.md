# Chokkhu-PyPi-Package

This repository contains the source code and documentation for the Chokkhu image classification Python package, which is available on PyPi. The Chokkhu is a powerful tool for classification image based on given data in google colab and jupyter notebook.

# How to run?

1. Create a virtual environment
```bash
    conda create -n Chokkhu python=3.8 -y
```
2. Activate this environment
```bash
    conda activate Chokkhu
```
3. Install required package
```bash
    pip install -r requirements_dev.txt
```
4. If do not build/install time generated metadata folder
```bash
    pip install -e .
    python setup.py develop
    python -m build
```