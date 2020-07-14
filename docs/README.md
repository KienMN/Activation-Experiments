# Activation Experiments

## Introduction
This repository contains source code of experiments using different activation functions.
## Requirements

- Python 3
- Tensorflow 2.*
- Tensorflow GPU (recommended)

## Installation
- Clone the repository and install the package
```
git clone https://github.com/KienMN/Activation-Experiments.git
cd Activation-Experiments
pip install -e .
```
- Install package using `pip`
```
pip install git+https://github.com/KienMN/Activation-Experiments.git
```

## Experiments
### Autoencoder
Run autoencoder experiment by the command
```
python tests/test_autoencoder.py --activation {activation_name} --dataset mnist --latent_dims 30 --n_epochs 50 --weights_dir tests/weights --losses_dir tests/losses
```