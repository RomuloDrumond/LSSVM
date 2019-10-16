# Overview

This repository contains a Jupyter Notebook with a python implementation of **Least Squares Support Vector Machine (LSSVM)** on **CPU** and **GPU**, you can find a bit of theory and the implementation on it. For a more enjoyable view of the notebook:
https://nbviewer.jupyter.org/github/RomuloDrumond/LSSVM/blob/master/LSSVM.ipynb

To install dependencies run `pip install -r requirements.txt` on the main directory.

### Important libraries used:

* Pandas, for loading and preprocessing of the data;
* Sklearn, for scaling features;
* Numpy, for matrices computation on CPU version;
* PyTorch, for matrices computations on GPU version;
* Scipy, for the fast `cdist` function;
