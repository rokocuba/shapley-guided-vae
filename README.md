# Shapley-Guided VAE: Dynamic Scaling of Reconstruction Error

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-latest-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **BSc in Computing, Thesis Project**  
> **Author:** Roko  
> **Institution:** Faculty of Electrical Engineering and Computing (FER), University of Zagreb  
---
## Abstract

This project investigates the optimization of Variational Autoencoder (VAE) training by replacing standard, fixed-hyperparameter loss scaling with a dynamic reweighting of per-feature reconstruction errors. By framing feature contributions as a cooperative game, the methodology leverages computationally efficient mathematical approximations of exponentially complex Shapley values to continuously adjust these weight factors during model training. Finally, the proposed approach is experimentally evaluated against static baselines to rigorously analyze the trade-off between the computational overhead of online Shapley estimation and resulting improvements in convergence speed and final reconstruction quality.

## Dataset

We use the **UCI Wine Quality Dataset**, which consists of 11 continuous physicochemical features (e.g., Fixed acidity, pH, Alcohol) and a quality score. The data is treated as an 11-dimensional continuous input for unsupervised reconstruction.

## Mathematical Framing

The core innovation treats VAE training as a cooperative game per epoch:
* **Players:** The 11 individual input features.
* **Payoff:** The reconstruction performance of the VAE given a specific subset (coalition) of features.

## References

[1] Anonymous, "When Encoders Should Stay Simple: An Empirical Analysis of Architectures for Variational Autoencoders," submitted for review at the *Int. Conf. on Learning Representations (ICLR)*, Dec. 2025. [Online]. Available: [OpenReview](https://openreview.net/forum?id=2hTLJEgCbv)

[2] A. A. Alemi, B. Poole, I. Fischer, J. V. Dillon, R. A. Saurous, and K. Murphy, "An information-theoretic analysis of deep latent-variable models," in *Proc. Int. Conf. on Learning Representations (ICLR)*, 2018. [Online]. Available: [OpenReview](https://openreview.net/pdf?id=H1rRWl-Cb)
