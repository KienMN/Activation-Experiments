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

## Main components
### Dynamic Parametric ReLU activation
This activation function is implemented as a subclass of `tf.keras.layers.Layer` so that is compatible with `tf.keras.Model`. The class name is `DPReLU` and can be found in `activation_layers/activation_layers.py`.

After installation, we can use this layer class as usual. For example,
```python
import tensorflow as tf
from activation_layers import DPReLU

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10),
  DPReLU()
])
```

### Layer replacement function
The `insert_layer_nonseq` function aims to replace layers in a model, even the model is not sequential. The implementation is in `activation_layers/activation_layers/utils.py`, following the answer in https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model. The following example shows how to replace layer with the name matching `'.*relu'` by a `DPReLU` layer.

```python
from activation_layers import DPReLU

def factory():
  return DPReLU(name='dprelu')

model = insert_layer_nonseq(model, '.*relu.*', factory, position='replace')
```

## Experiments
### Autoencoder
Run autoencoder experiment by the command (see the full list of arguments in the file `tests/test_autoencoder.py`).
```
python tests/test_autoencoder.py --activation {activation_name} --dataset mnist --latent_dims 30 --n_epochs 50
```

### ResNet50
Run image classification using ResNet50 model experiment by the command (see the full list of arguments in the file `tests/test_resnet.py`).
```
python tests/test_resnet.py --activation {activation_name} --dataset cifar10 --epochs 50 --batch_size 128
```