# Lattice Spectral Dynamics (LSD): A Non-Hermitian Approach to Vacuum Stability

**LSD** is a high-performance PyTorch framework for simulating non-equilibrium field dynamics on 4-dimensional lattices. It explores the emergence of stable coupling constants from a scalar field evolving under a non-Hermitian Hamiltonian with spectrally structured noise.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Overview

Standard Lattice Gauge Theory typically employs Monte Carlo methods to sample equilibrium configurations. This project takes a different approach: it models the vacuum as an open quantum system subject to a specific class of stochastic driving forces derived from Random Matrix Theory (GUE statistics).

The core hypothesis is that dimensionless constants (such as the fine-structure coupling $\alpha$) appear as **asymptotic fixed points** of the system's relaxation dynamics when the driving noise spectrum matches the zeros of the Riemann Zeta function.

### Key Features

*   **4D Hypercubic Lattice:** Optimized tensor operations for 4-dimensional field evolution.
*   **Non-Hermitian Dynamics:** Implements a modified diffusion-reaction equation with complex phase rotation.
*   **Spectral Noise Injection:** A custom kernel that synthesizes stochastic kicks $\theta(t)$ based on the imaginary parts of the Riemann Zeta zeros.
*   **GPU Acceleration:** Fully vectorized PyTorch implementation for CUDA-enabled devices.

## The Physics

The simulation evolves a complex scalar field $\psi(x,t)$ according to a discrete update rule:

$$ \psi_{t+1} = \mathcal{D}[\psi_t] \cdot e^{i \omega_{\text{eff}} \Delta t} \cdot e^{i \theta(t)} $$

Where:
*   $\mathcal{D}$ is a 4D discrete Laplacian operator (modeling diffusion/kinetic energy).
*   $\omega_{\text{eff}}$ is a local phase rotation dependent on the field gradient $|\nabla \psi|^2$ (modeling self-interaction/gravity).
*   $\theta(t)$ is the "Spectral Noise" term, structured according to the GUE statistics of the Zeta function.

We monitor the **Emergent Coupling Ratio**, defined as the ratio of the interaction energy (field-noise coupling) to the geometric tension (field gradient energy). Preliminary runs suggest this ratio stabilizes to values consistent with standard physical constants.

## Installation

```bash
git clone https://github.com/yourusername/lattice-spectral-dynamics.git
cd lattice-spectral-dynamics
pip install -r requirements.txt
