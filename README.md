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

We use the **UCI Wine Quality Dataset**, which consists of 11 continuous physicochemical features (e.g., Fixed acidity, pH, Alcohol) and a quality score. The data is treated as an 11-dimensional continuous input for unsupervised reconstruction.
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
_Modeling wine preferences by data mining from physicochemical properties._
Decision Support Systems, 47(4):547–553, 2009.

Dataset available from the UCI Machine Learning Repository.

## Mathematical Framing

The core innovation treats VAE training as a cooperative game per epoch:

- **Players:** The 11 individual input features.
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
