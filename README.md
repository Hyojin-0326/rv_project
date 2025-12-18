# Error-Driven Importance Modeling for Efficient 3D Gaussian Splatting

This repository contains the official implementation of **Error-Driven Importance Modeling for Efficient 3D Gaussian Splatting**, a research project that improves densification and pruning strategies in 3D Gaussian Splatting (3DGS) by introducing explicit error signals, Gaussian tagging, and a learnable importance score decoupled from opacity.

The goal of this project is to **preserve fine geometric and appearance details while reducing unnecessary memory usage**, addressing key limitations of heuristic-based densification and opacity-driven pruning in the original 3DGS pipeline.

---

## Overview

The proposed method extends the original 3D Gaussian Splatting framework with three core components:

1. **Per-pixel error–driven densification**
   Gaussians are refined based on explicit rendering error signals instead of indirect gradient heuristics.

2. **Gaussian tagging (lifespan-aware pruning)**
   Each Gaussian tracks its creation and survival history to prevent premature pruning.

3. **Learnable importance score (`s_k`)**
   A lightweight MLP predicts Gaussian importance independently of opacity, enabling robust pruning even in low-opacity or transparent regions.

These components work together to allocate model capacity where it matters most, improving perceptual quality and efficiency.

---

## Key Contributions

* **Error-driven densification** using per-pixel rendering error extracted via an auxiliary loss without additional backward passes
* **Adaptive thresholding** based on error-score distributions to reduce scene-specific hyperparameter tuning
* **Gaussian tagging** to avoid underdensification caused by aggressive early pruning
* **MLP-based importance modeling** that decouples pruning decisions from opacity
* **Importance-modulated rendering** that suppresses noise while preserving thin or semi-transparent structures

---

## Results
<img width="789" height="259" alt="image" src="https://github.com/user-attachments/assets/98729cdb-d3c2-4f2b-a51a-efe104e6e18f" />
<img width="473" height="628" alt="image" src="https://github.com/user-attachments/assets/60f201cd-9850-40b3-8388-a31b67b64e24" />
<img width="469" height="632" alt="image" src="https://github.com/user-attachments/assets/adeb99bc-b0a6-4ecc-8e5e-1750cfd8748d" />



---

## Method Summary

### 1. Per-Pixel Error Modeling (`E_k`)

* Computes per-Gaussian error scores derived from pixel-wise reconstruction error
* Error is accumulated across views and summarized using a max or statistics-based formulation
* Implemented efficiently by scaling an auxiliary loss term and recovering error from gradients
* Used to guide **densification (splitting / cloning)** in high-error regions

### 2. Gaussian Tagging

* Each Gaussian maintains lightweight metadata indicating its **generation and survival duration**
* Pruning is conditioned not only on score thresholds but also on lifespan
* Prevents newly spawned Gaussians from being removed before meaningful optimization

> Note: Gaussian tagging logic is implemented directly in the training and model code. While not fully formalized in the report, the implementation follows a lifespan-aware pruning rule inferred from creation iteration and survival stability.

### 3. Importance Score Modeling (`s_k`)

* Each Gaussian is assigned a learnable importance score predicted by an MLP
* Importance is **independent of opacity** and learned through reconstruction loss
* Effective opacity during rendering is modulated as:

```
α_eff = α · s_k
```

* Includes regularization to avoid trivial collapse (all scores → 1)
* Enables **importance-only pruning** without relying on opacity thresholds

---

## Repository Structure

```
.
├── train.py                # Training loop with error extraction, tagging, and pruning
├── gaussian_model.py       # Core Gaussian representation and importance modeling
├── subfunction.py          # Helper utilities (error stats, scheduling, etc.)
├── arguments/              # Training and model configuration
├── utils/                  # Rendering, loss, and math utilities (from 3DGS base)
├── output/                 # Training outputs and checkpoints
└── README.md
```

This codebase is built on top of the official **GRAPHDECO 3D Gaussian Splatting implementation**.

---

## Experimental Setup

* **Datasets**: Bicycle, Flowers, Garden
* **Metrics**: PSNR, SSIM, LPIPS, model size
* **Baseline**: Original 3D Gaussian Splatting (Kerbl et al., 2023)

All experiments follow the same training protocol as the baseline unless otherwise stated.

---

## Results Highlights

* Improved preservation of **fine details and thin structures**
* Significant reduction of **low-opacity noise artifacts**
* More selective pruning leading to **smaller model size** with comparable or improved perceptual quality
* Especially strong qualitative gains on **Flowers**, where opacity-based pruning is unreliable

---

## Limitations

* Still depends on a small number of hyperparameters (e.g., importance regularization strength)
* Limited evaluation on large-scale datasets due to GPU constraints
* Full generalization across diverse scenes remains to be validated

---

## Future Work

* Importance-only pruning without any opacity-based heuristics
* Fully adaptive controllers (e.g., quantile-based thresholds) to eliminate manual tuning
* Large-scale and cross-domain evaluation
* Integration with compression and quantization pipelines

---

## License

This project is based on the official 3D Gaussian Splatting code released by GRAPHDECO under a **research-only license**.
Please refer to the original repository for licensing details.

---

## References

* Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, TOG 2023
* Rota Bulò et al., *Revising Densification in Gaussian Splatting*, arXiv 2024

---

## Authors

* Seoyeon Kim
* Hyojin Kwon

Robot Vision Project, Dec 2025
