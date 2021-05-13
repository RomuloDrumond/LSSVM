# Overview

This repository contains a python implementation of the **Least Squares Support Vector Machine (LSSVM)** model on **CPU** and **GPU**, you can find a bit of theory and usage of the code on the LSSVC.ipynb jupyter notebook. For a more enjoyable view of the notebook:
https://nbviewer.jupyter.org/github/RomuloDrumond/LSSVM/blob/master/LSSVC.ipynb

To install dependencies run `pip install -r requirements.txt` on the main directory.

### Important libraries used:

* Pandas, for loading and preprocessing of the data;
* Sklearn, for scaling features;
* Numpy, for matrices computation on CPU version;
* PyTorch, for matrices computations on GPU version;
* Scipy, for the fast `cdist` function;
