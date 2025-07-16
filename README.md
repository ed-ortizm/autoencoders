# Variational Autoencoders for SDSS Spectra

## Overview

This repository contains the full pipeline for building, training, and analyzing **Variational Autoencoders (VAEs)** using galaxy spectra from the **Sloan Digital Sky Survey (SDSS)**. VAEs are generative models capable of learning compact representations (latent space) of complex data distributions. In this project, VAEs are applied to rest-frame SDSS spectra to enable anomaly detection, synthetic data generation, and latent space analysis.

The implementation includes support for different divergence regularizations (e.g., **Kullback-Leibler Divergence (KLD)** and **Maximum Mean Discrepancy (MMD)**), hyperparameter search, reconstruction evaluation, and visualization of latent embeddings via **t-SNE** and **UMAP**.

## Features

* 🧠 **Flexible VAE Architecture**: Modular implementation of autoencoders with customizable latent dimensions and loss regularization.
* 🧪 **Divergence Control**: Includes KLD and MMD as regularization options.
* 🔍 **Latent Space Analysis**: Visualizations with t-SNE and UMAP, as well as pair plots and clustering insights.
* ⚡ **Reconstruction Scoring**: Tools to evaluate model performance via reconstruction metrics.
* 📈 **Training Utilities**: Training scripts with support for local and remote environments, plus tracking of training history.

## Getting Started

### Prerequisites

* Python 3.8+
* TensorFlow
* NumPy, Matplotlib, Scikit-learn
* UMAP, t-SNE, Seaborn, etc.

Install dependencies:

```bash
pip install -r requirements.txt
```

### Repository Structure

```bash
├── divergence/              # KLD and MMD metric computation and visualization
├── latent/                  # Latent space analysis: tsne, umap, pair plots
├── speed/                   # Scripts for measuring reconstruction speed
├── src/
│   └── autoencoders/        # Core VAE module
│       ├── ae.py
│       ├── divergence/
│       ├── hyperSearch.py
│       └── plotAE.py
├── training/                # Training and hyperparameter configuration
│   ├── train.py
│   ├── history.py
│   └── hyperSearch.py
├── LICENSE
├── pyproject.toml
├── README.md
```

## Usage

1. **Training a VAE:**
   Configure your training parameters in `training/train.ini` and run:

   ```bash
   python training/train.py --config training/train.ini
   ```

2. **Compute Divergence Terms:**
   Choose between KLD or MMD and use scripts in the `divergence/` folder to evaluate:

   ```bash
   python divergence/kld.py --config divergence/kld.ini
   ```

3. **Latent Space Visualization:**

   ```bash
   python latent/umap_visual.py --config latent/umap_visual.ini
   ```

4. **Reconstruction Evaluation:**
   Evaluate reconstruction performance using:

   ```bash
   python speed/reconstruction.py --config speed/reconstruction.ini
   ```

5. **Hyperparameter Search:**

   ```bash
   python training/hyperSearch.py --config training/hyperSearch.ini
   ```

## Contact

For questions or feedback, please contact:

* Edgar Ortiz ([ed.ortizm@gmail.com](mailto:ed.ortizm@gmail.com))
* Mederic Boquien ([mederic.boquien@oca.eu)](mailto:mederic.boquien@oca.eu))