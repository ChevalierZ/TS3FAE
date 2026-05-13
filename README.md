# TS<sup>3</sup>FAE: Superpixel-Based Two Stream Spatial Spectral Fusion Autoencoder for Hyperspectral Target Detection

Official PyTorch implementation of the paper:

> **Superpixel-Based Two Stream Spatial Spectral Fusion Autoencoder for Hyperspectral Target Detection**

---

## Authors

- Weibo Zhang
- Xiaobian Wu *(Corresponding Author)*
- Zhonghao Chen
- Yakun Wang
- Degang Wang
- Hongqi Zhang

### Affiliations

- Nanjing Hydraulic Research Institute, Nanjing, China
- Hohai University, Nanjing, China
- School of Technology, Beijing Forestry University, Beijing, China
- Aerospace Information Research Institute, Chinese Academy of Sciences, Beijing, China


---

## Abstract

Hyperspectral remote sensing images (HSI), with their integrated spatial–spectral characteristics, offer rich information for target detection. However, the common limitation of having only a single target spectrum in most datasets restricts the effective use of spatial context, which is vital for addressing spectral uncertainty. To overcome this challenge, a superpixel-based patch construction strategy combined with a two-stream spatial–spectral fusion autoencoder (TS<sup>3</sup>FAE) is proposed.

A neighborhood spectral patch construction module organizes pixels into spatial–spectral patches reflecting real-world spatial distributions, enabling effective training of target and background representations from limited spectral samples. The model leverages spatial–spectral joint convolution and autoencoding to jointly learn spectral and local spatial features within superpixel neighborhoods. Additionally, a linear background suppression module enhances target–background separability.

Experiments on four benchmark datasets demonstrate that TS<sup>3</sup>FAE outperforms existing methods.

---

## Keywords

- Hyperspectral target detection
- Superpixel segmentation
- Spectral-spatial information fusion

---
  
# Framework Overview

TS<sup>3</sup>FAE mainly consists of:

1. **Superpixel-based neighborhood spectral patch construction**
2. **Two-stream spatial-spectral fusion autoencoder**
3. **Linear background suppression module**

The proposed framework jointly learns spectral and local spatial representations for hyperspectral target detection under limited target samples.

---

# Project Structure

```bash
TS3FAE/
│
├── exp.py                     # Main training/testing script
├── model.py                   # TS3FAE model definition
├── loss.py                    # Loss functions
├── GBIS.py                    # Graph-Based Image Segmentation
├── utils.py                   # Utility functions
│
├── data/
│   ├── Sandiego.mat
│   ├── Cat Island.mat
│   ├── TexasCoast.mat
│   └── Bay Champagne.mat
│
├── checkpoints/
│   ├── b_model_Sandiego.pth
│   ├── t_model_Sandiego.pth
│   ├── b_model_CatIsland.pth
│   ├── t_model_CatIsland.pth
│   ├── b_model_TexasCoast.pth
│   ├── t_model_TexasCoast.pth
│   ├── b_model_BayChampagne.pth
│   └── t_model_BayChampagne.pth
│
└── README.md
```
---
# ⚙️ Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.10
---
# Datasets

The experiments are conducted on four benchmark hyperspectral target detection datasets:

| Dataset | Format |
|---|---|
| Sandiego | `.mat` |
| Cat Island | `.mat` |
| TexasCoast | `.mat` |
| Bay Champagne | `.mat` |

# Usage

## Train / Test

Run the main script:

```bash
python exp.py
```

You can modify dataset paths and hyperparameters in `exp.py`.

---

# Pretrained Models

We provide pretrained model weights for all four datasets.

Each dataset contains:

- `b_model_*.pth` : background branch model
- `t_model_*.pth` : target branch model

Example:

```bash
b_model_Sandiego.pth
t_model_Sandiego.pth
```

---

# Notes

- The implementation is based on PyTorch.
- GBIS is used for superpixel segmentation and neighborhood construction.
- All datasets should be stored in MATLAB `.mat` format.

---
---

# License

This project is released for academic research purposes only.

---

# Acknowledgement

We sincerely thank the providers of the public hyperspectral datasets and the related open-source research community.
