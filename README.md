# Automated Sleep Spindles and K-complexes Detection in FAME

This document describes the three reference implementations stored in this directory and explains how to adapt them to a private dataset such as FAME. The original model code should be kept unchanged whenever possible; dataset-specific work should be isolated in loaders, conversion scripts, and configuration files.

## Contents

- [Projects](#projects)
- [Setup](#setup)
- [Dataset](#Dataset)

## Projects

1. SEED
- **Original Repository:** [Sleep-EEG-Event-Detector](https://github.com/nicolasigor/Sleep-EEG-Event-Detector)
- **What we provide:** Custom dataset classes for the FAME dataset:
  - `fame_ss.py`: Dataset adapter for Sleep Spindles detection.
  - `fame_kc.py`: Dataset adapter for K-complexes detection.
- **Integration:** SEED is a TensorFlow-based event detector. To use our FAME dataset, clone the original repository, place our dataset classes into the `sleeprnn/data/` directory, and register them in the dataset reader. 

2. DOSED
- **Original Repository:** [DOSED](https://github.com/Dreem-Organization/dosed/) 
- **What we provide:** 
  - `prepare_fame.py`: The preprocessing script that reads raw FAME EDF/annotation files, applies bandpass filtering and resampling, and saves the outputs as `.npy` and `.json` files.
  - `fame_data.py`: The data loading utility that handles subject splits, stage filtering, and event loading for the DOSED model.
- **Integration:** You need to use our provided scripts to generate one HDF5 file per recording, and then use the original DOSED repository's notebooks for training and evaluation.

3. U-Net
- **Original Repository:** [U-Net](https://github.com/LordXX505/SpindleDetection) 
- **What we provide:** `get_pre_data_FAME.py`, a standalone custom data preprocessing script for the FAME dataset. It reads raw EDFs and annotations, applies resampling (256Hz), and generates the fixed-length `.npy` signal and label arrays required by the model.
- **Integration:** To use this dataset adaptation, first clone the original SpindleU-Net repository. Run our `get_pre_data_FAME.py` script, which will automatically process the raw FAME data and save the `epoch_*.npy` and `label_*.npy` arrays into the `SpindleData/FAME` or `KcomplexData/FAME` directory. After the data is generated, you can run the original `SpindleU_Net_pre_train.py` keeping the core network and training logic completely unchanged.


## Setup

Because this repository only provides dataset adapters and preprocessing scripts for the FAME dataset, **you must clone the original repositories** to run the complete models. 

We highly recommend using a separate Conda environment for each project, as their framework and dependency requirements may conflict.

 1. SEED
git clone https://github.com/nicolasigor/Sleep-EEG-Event-Detector.git
cd Sleep-EEG-Event-Detector
conda create -n seed python=3.9
conda activate seed
pip install -r seed/Sleep-EEG-Event-Detector-main/requirements.txt

 2. Dosed
git clone https://github.com/Dreem-Organization/dosed.git
cd dosed
conda create -n dosed python=3.9
conda activate dosed
pip install -r requirements.txt

 3. U-net
git clone https://github.com/LordXX505/SpindleDetection.git
cd SpindleDetection
conda create -n SpindleDetection
conda activate SpindleDetection
pip install -r requirements.txt


## Dataset
We use the FAME datasets.
You can download from https://doi.org/10.6084/m9.figshare.33979861
