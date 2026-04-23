# Benchmark RST

This repository contains the modified RootSystemTracker pipeline used in the study: **"Do Segmentation Objectives Match Reconstruction Goals? An Epoch-Wise Analysis of Root Phenotyping Pipelines"**.

This code is part of an epoch-wise benchmarking framework designed to evaluate how segmentation training choices impact downstream deterministic graph reconstruction. And which segmentation metrics best predict physical trait extraction performance.

## Dataset

This pipeline was trained and evaluated on the **HIRROS dataset**.

* Download the dataset from [Zenodo](https://doi.org/10.5281/zenodo.19663614).

## Pipeline Modifications & Features

The original RootSystemTracker relied on a spatiotemporal mean shift rupture detection on raw images. For this benchmark, we introduced a Deep Learning segmentation module:

* **Architectures:** Integrated SegFormer (MIT-B0) and U-Net (EfficientNet-B0) encoders.
* **Loss Functions:** Models can be trained using BCE, Dice, and Dice + clDice.
* **Post-processing:** The spatiotemporal mean shift (Rupture Down Detector) and a max slope detector (both with a 0.75 threshold) are applied to the generated probability heatmaps instead of raw images to assemble a `date_map`.
* **Graph Extraction & Matching:** The `date_map` is ingested by the RST Java pipeline to extract the spatiotemporal RSML. Plant-to-plant matching is handled using Euclidean distance for seeds and Hausdorff distance for root curves.

## Usage & Reproducibility

*All hyperparameters were kept fixed to isolate the effect of the loss functions.*

1. **Training:** Run the segmentation training script. Checkpoints are saved at every epoch. Inputs size is 512x512
2. **Reconstruction:** Run the inference and graph extraction script on the saved checkpoints. Deterministic seeds are fixed across all runs.

## Folder content

The code is organized as follows:

* The ``CreateRSADataset`` folder contains code that was used to format RSML files from RootsystemTracker software into more standart RSML files (e.g. time attribute becomes a function).
* The ``RootSystemTracker`` folder is a link to a fork from RootSystemTracker method [Fernandez et al., 2022]. The main script ``main`` function was changed in order to compute inferences from input ``date_map``.
* The ``RSA_deep_working`` folder contains the training data, json configuration files and codes for dataloaders, loss functions, segmentation metrics, model implementation, training and evaluating the models.
* The ``RSA_reconstruction`` folder contains a copy of the codes for dataloaders and segmentation models (for inference) and codes for measuring root traits on MTGs, for reconstructing RSA from images and to make automatic measurements.
* The ``Scripts`` folder contains the scripts that were used to execute the code in Ad Astra, Jean Zay or Grid5000.

## Citation

This paper is currently under review. If you use this code or dataset, please cite:

>  Gandeel, L., Pradal, C., Akbarinia, R., & Fernandez, R. (202X). Do Segmentation Objectives Match Reconstruction Goals? An Epoch-Wise Analysis of Root Phenotyping Pipelines. *[xx, xx(x), pp. xx-xx]*.
