import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from autoencoder_models import *
from activation_layers import modified_he_normal, dprelu_normal, prelu_normal, xavier_untruncated_normal

def get_parser():
  """
  Generate a parameters parser.
  """
  parser = ArgumentParser(description='Autoencoder experiments')
  
  # Activation functions
  parser.add_argument(
    '--activation',
    type=str,
    choices=['relu', 'prelu', 'frelu', 'bn_relu', 'dprelu'],
    required=True,
    help="Activation")

  parser.add_argument(
    '--alpha',
    type=float,
    default=None
  )

  parser.add_argument(
    '--beta',
    type=float,
    default=None
  )

  # Dataset
  parser.add_argument(
    '--dataset',
    type=str,
    choices=['mnist', 'cifar10', 'cifar100'],
    default='mnist',
    help='Name of dataset')

  # Directory to save results
  parser.add_argument(
    '--weights_dir',
    type=str,
    default=None,
    help='Directory to save weights')

  parser.add_argument(
    '--losses_dir',
    type=str,
    default=None,
    help='Directory to save losses')

  # Training parameters
  parser.add_argument(
    '--latent_dim',
    type=int,
    default=30,
    help='Number of dimensions of latent space')
  
  parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help='Batch size')

  parser.add_argument(
    '--n_epochs',
    type=int,
    default=50,
    help='Number of epochs')

  parser.add_argument(
    '--learning_rate',
    type=float,
    default=1e-3,
    help='Learning rate')

  # Weight initialization
  parser.add_argument(
    '--weight_initialization', '-wi',
    type=str,
    choices=['xavier', 'he', 'modified_he', 'dprelu', 'prelu', 'xavier_untruncated'],
    default='xavier'
  )

  return parser

def run(args):
  # Directory of this file
  work_dir = os.path.abspath(os.path.dirname(__file__))
  
  # Prepare the dataset
  if args.dataset == 'mnist':
    IMG_SHAPE = [28, 28, 1]
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape([-1, 28, 28, 1]).astype('float32')
    test_images = test_images.reshape([-1, 28, 28, 1]).astype('float32')

  elif args.dataset == 'cifar10':
    IMG_SHAPE = [32, 32, 3]
    (train_images, _), (test_images, _) = tf.keras.datasets.cifar10.load_data()
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
  
  elif args.dataset == 'cifar100':
    IMG_SHAPE = [32, 32, 3]
    (train_images, _), (test_images, _) = tf.keras.datasets.cifar100.load_data()
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

  train_images /= 255.
  test_images /= 255.

  # Training parameters
  latent_dim = args.latent_dim
  batch_size = args.batch_size
  epochs = args.n_epochs
  learning_rate = args.learning_rate

  # Model
  # model_name = 'Autoencoder_with_{}_on_{}'.format(args.activation, args.dataset)
  model_name = 'autoencoder_{}_on_{}_{}_init'.format(
    args.activation,
    args.dataset,
    args.weight_initialization
  )

  if args.activation == 'dprelu':
    model_name += '_alpha_{}_beta_{}'.format(args.alpha, args.beta)
  elif args.activation == 'prelu':
    model_name += '_alpha_{}'.format(args.alpha)

  print(model_name)
  
  if args.dataset == 'mnist':
    if args.activation == 'bn_relu':
      model = AutoEncoderWithBatchNormReLU(
        input_dims=IMG_SHAPE,
        latent_dim=latent_dim)
    else:
      model = MnistAutoencoder(
        input_dims=IMG_SHAPE,
        latent_dim=latent_dim,
        activation=args.activation,
        alpha=args.alpha,
        beta=args.beta,
        shared_axes=None,
        weight_initialization=args.weight_initialization)
  elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
    model = Cifar10ConvAutoencoder(
      input_dims=IMG_SHAPE,
      latent_dim=latent_dim,
      hidden_dim=1024,
      activation=args.activation,
      alpha=args.alpha,
      beta=args.beta,
      weight_initialization=args.weight_initialization)

  # Check the value of initialized-parameters of DPReLU
  # for layer in model.encoder.layers:
  #   # print(layer.name)
  #   if 'dp_re_lu' in layer.name:
  #     print(layer.weights)

  # Check the histogram of weight intialization
  # y = model.encoder.layers[0].weights[0].numpy().reshape(-1,1)
  # print(y)

  # fig, ax = plt.subplots(1, 1)
  # ax.hist(y)
  # ax.set_title('After')
  # plt.show()

  model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(learning_rate))
  
  # print(model.encoder.summary())
  # print(model.decoder.summary())

  history = model.fit(
    x=train_images,
    y=train_images,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(test_images, test_images))

  # Save the loss
  filename = model_name + '_loss.csv'
  pd.DataFrame(history.history).to_csv(filename, index=False)

if __name__ == '__main__':
  # Generate parser
  parser = get_parser()
  args = parser.parse_args()

  # Run experiment
  run(args)