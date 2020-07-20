from __future__ import print_function, division, absolute_import

import os
import re
import tensorflow as tf
import pandas as pd
from argparse import ArgumentParser
from activation_layers import DPReLU
from activation_layers.utils import insert_layer_nonseq
from tensorflow.keras.models import load_model

work_dir = os.path.abspath(os.path.dirname(__file__))

parser = ArgumentParser()
parser.add_argument('--filepath', '-p', type=str)
parser.add_argument('--model_name', '-n', type=str)

args = parser.parse_args()

dprelu_weights = {
  'layer': [],
  'alpha': [],
  'threshold': [],
  'beta': [],
  'bias': []
}

model_filepath = args.filepath
model_name = args.model_name

model = load_model(model_filepath, custom_objects={'DPReLU': DPReLU})

for layer in model.layers:
  if re.match('.*dprelu.*', layer.name):
    alpha = layer.weights[0].numpy().mean()
    threshold = layer.weights[1].numpy().mean()
    beta = layer.weights[2].numpy().mean()
    bias = layer.weights[3].numpy().mean()
    
    dprelu_weights['layer'].append(layer.name)
    dprelu_weights['alpha'].append(alpha)
    dprelu_weights['threshold'].append(threshold)
    dprelu_weights['beta'].append(beta)
    dprelu_weights['bias'].append(bias)

dprelu_weights = pd.DataFrame(dprelu_weights)
dprelu_weights.to_csv(work_dir + '/{}_dprelu_weights.csv'.format(model_name), index=False)