# Project for Deep Learning

Final project for the deep learning course. 

## RMFD

Multi-label disease for detection using Retina Fundus Images.


## Installation

Dowload the data from: https://riadd.grand-challenge.org/download-all-classes/ 

Run the following command to generate the numpy binaries of all datasets:

```bash
Python image_preprocessing_windows.py
```

To run with 100 samples run the following command:

```bash
Python rfmd_100.py
```

To train with all 949 samples, run the following command:

```bash
Python rfmd.py
```

