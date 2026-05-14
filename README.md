# Shapley-Guided VAE: Dynamic Scaling of Reconstruction Error

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-latest-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **BSc in Computing, Thesis Project**  
> **Author:** Roko Čubrić\
> **Institution:** Faculty of Electrical Engineering and Computing (FER), University of Zagreb\
> **Mentor:** Tomislav Burić

---

## Abstract

This project investigates the optimization of Variational Autoencoder (VAE) training by replacing standard, fixed-hyperparameter loss scaling with a dynamic reweighting of per-feature reconstruction errors. By framing feature contributions as a cooperative game, the methodology leverages computationally efficient mathematical approximations of exponentially complex Shapley values to continuously adjust these weight factors during model training. Finally, the proposed approach is experimentally evaluated against static baselines to rigorously analyze the trade-off between the computational overhead of online Shapley estimation and resulting improvements in convergence speed and final reconstruction quality.

## Dataset

We use the **UCI Multiple Features (mfeat) Dataset**.

For this project, the relevant notion of class is the **six input classes** (descriptor groups) that are concatenated into one flat VAE input vector. Each sample contains all six input classes.

| Dataset property     | Value |
| -------------------- | ----: |
| Total samples        |  2000 |
| Input classes        |     6 |
| Train samples        |  1600 |
| Test samples         |   400 |
| Total input features |   649 |

| Input class | Description                | Floats |
| ----------- | -------------------------- | -----: |
| `fou`       | Fourier coefficients       |     76 |
| `fac`       | Profile correlations       |    216 |
| `kar`       | Karhunen-Love coefficients |     64 |
| `pix`       | Pixel averages             |    240 |
| `zer`       | Zernike moments            |     47 |
| `mor`       | Morphological features     |      6 |

Dataset available from the UCI Machine Learning Repository.

## Mathematical Framing

The core innovation treats VAE training as a cooperative game per epoch:

- **Players:** The input features.
- **Payoff:** The reconstruction performance of the VAE given a specific subset (coalition) of features.

## Requirements

- Python 3.11+
- (Recommended) `uv` package manager
- Alternatively: standard `pip`

---

## Installation

### Option A — Recommended (uv)

```bash
uv venv
# activate the environment
# CPU-only (default for most users)
uv sync --extra cpu

# CUDA 12.1
uv sync --extra cu121
```

---

### Option B — Standard pip

```bash
python -m venv .venv
# activate the environment
pip install -r requirements.txt
```

## References

- Hugh Chen, Ian C. Covert, Scott M. Lundberg, and Su-In Lee. _Algorithms to estimate Shapley value feature attributions._ arXiv:2207.07605, 2022. https://doi.org/10.48550/arXiv.2207.07605
