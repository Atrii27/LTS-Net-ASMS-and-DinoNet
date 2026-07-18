# Landslide Segmentation using ASDMS U-Net and DINOv2 + ASDMS U-Net

This repository contains two deep learning pipelines for multispectral landslide segmentation using satellite imagery. Both models are implemented as Jupyter notebooks and are located in the `code/` directory.

## Repository Structure

```text
project/
│
├── code/
│   ├── asdms.ipynb
│   └── asdms_dino.ipynb
│
├── data/
│   ├── 20220503_050008_52_2474_3B_AnalyticMS_8b_clip.tif
20241110_053942_45_24f7_3B_AnalyticMS_SR_8b_clip_Hybrid_mask.tif
│   ├── 20241110_053942_45_24f7_3B_AnalyticMS_SR_8b_clip.tif│
│
│
└── README.md
```

---

# 1. `asdms.ipynb`

## Description

This notebook implements the **ASDMS U-Net** architecture for binary landslide segmentation from multispectral satellite imagery.

The workflow includes:

* Loading GeoTIFF satellite images
* Image normalization
* Binary mask generation
* Patch extraction
* Training on the 2024 dataset
* Validation
* Whole-image prediction on the 2022 dataset
* Quantitative evaluation
* Visualization of predictions

## Features

* TensorFlow/Keras implementation
* Patch-based training
* Binary Cross Entropy loss
* Full-image reconstruction from predicted patches
* Evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Dice coefficient
  * IoU
* Confusion matrix
* Radar plot
* Performance comparison plots

## Training Dataset

* **2024 multispectral satellite image**
* Corresponding hybrid landslide mask

## Testing Dataset

* **2022 multispectral satellite image**
* Ground-truth landslide mask

---

# 2. `asdms_dino.ipynb`

## Description

This notebook implements a **DINOv2 + ASDMS U-Net** hybrid architecture using PyTorch.

Instead of learning features from scratch, DINOv2 is used as a pretrained Vision Transformer encoder to extract rich semantic representations, which are then decoded using an ASDMS-inspired decoder with deep supervision.

The workflow includes:

* Loading GeoTIFF images
* Patch extraction
* Data augmentation
* DINOv2 feature extraction
* Multi-layer feature fusion
* ASDMS decoder
* Deep supervision
* Whole-image inference on the 2022 dataset
* Quantitative evaluation
* Visualization

## Architecture

```
Input (4-band multispectral image)
        │
Input Adapter (4 → 3 channels)
        │
Pretrained DINOv2 Encoder
        │
Multi-layer Feature Fusion
        │
ASDMS Decoder
        │
Deep Supervision
        │
Binary Segmentation Mask
```

## Features

* PyTorch implementation
* Pretrained DINOv2 encoder
* Multi-scale transformer feature fusion
* Deep supervision
* Dice + BCE loss
* AdamW optimizer
* Cosine Annealing scheduler
* Whole-image patch inference
* Evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Dice coefficient
  * IoU
* Confusion matrix
* Prediction overlay

---

# Dataset

The experiments use multispectral PlanetScope imagery.

## Training

* 2024 multispectral satellite image
* Corresponding hybrid landslide mask

## Testing

* 2022 multispectral satellite image
* Ground-truth landslide mask

---

# Input Data

Each satellite image contains four spectral bands:

1. Blue
2. Green
3. Red
4. Near Infrared (NIR)

The segmentation task is binary:

* 0 → Background
* 1 → Landslide

---

# Software Requirements

* Python 3.10+
* TensorFlow (for ASDMS)
* PyTorch (for DINOv2 + ASDMS)
* timm
* rasterio
* NumPy
* matplotlib
* scikit-learn
* albumentations
* torchmetrics
* seaborn
* tqdm

---

# Output

The notebooks generate:

* Trained model weights
* Predicted segmentation masks
* Overlay visualizations
* Confusion matrices
* Performance metrics
* Dice score
* IoU score
* Precision
* Recall
* F1-score
* Performance summary plots

---

# Notes

* Both notebooks use patch-based processing for efficient handling of large GeoTIFF images.
* The ASDMS notebook is implemented using TensorFlow/Keras.
* The DINOv2 + ASDMS notebook is implemented using PyTorch and leverages a pretrained Vision Transformer encoder.
* The same evaluation protocol is used for both models to enable a fair comparison.

---

# Citation

If you use this repository in your research, please cite the corresponding publication or acknowledge this implementation in your work.
