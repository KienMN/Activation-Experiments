import os
import time
import json
import pandas as pd
from argparse import ArgumentParser
from autoencoder_models import *

def get_parser():
  """
  Generate a parameters parser.
  """
  parser = ArgumentParser(description='Autoencoder experiments')
  
  # Activation functions
  parser.add_argument('--activation', type=str, choices=['relu', 'prelu', 'frelu', 'bn_relu', 'dprelu'], required=True, help="Activation")

  # Dataset
  parser.add_argument('--dataset', type=str, choices=['mnist'], default='mnist', help='Name of dataset')

  # Directory to save results
  parser.add_argument('--weights_dir', type=str, default=None, help='Directory to save weights')
  parser.add_argument('--losses_dir', type=str, default=None, help='Directory to save losses')

  # Training parameters
  parser.add_argument('--latent_dim', type=int, default=30, help='Number of dimensions of latent space')
  parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
  parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs')
  parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')

  return parser

def run(args):
  # Directory of this file
  work_dir = os.path.abspath(os.path.dirname(__file__))
  
  # Prepare the dataset
  if args.dataset == 'mnist':
    input_shape = [28, 28, 1]
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape([-1, 28, 28, 1]).astype('float32')
    test_images = test_images.reshape([-1, 28, 28, 1]).astype('float32')

    train_images /= 255.
    test_images /= 255.

  # Training parameters
  latent_dim = args.latent_dim
  batch_size = args.batch_size
  epochs = args.n_epochs
  learning_rate = args.learning_rate

  # Model
  model_name = 'Autoencoder_with_{}_on_{}'.format(args.activation, args.dataset)
  print(model_name)
  if args.activation == 'relu':
    model = AutoEncoderWithReLU(input_dims=input_shape, latent_dim=latent_dim)
  elif args.activation == 'prelu':
    model = AutoEncoderWithPReLU(input_dims=input_shape, latent_dim=latent_dim)
  elif args.activation == 'frelu':
    model = AutoEncoderWithFReLU(input_dims=input_shape, latent_dim=latent_dim)
  elif args.activation == 'bn_relu':
    model = AutoEncoderWithBatchNormReLU(input_dims=input_shape, latent_dim=latent_dim)
  elif args.activation == 'dprelu':
    model = AutoEncoderWithDPReLU(input_dims=input_shape, latent_dim=latent_dim)
  
  model.compile(loss='mse',
                optimizer=tf.keras.optimizers.Adam(learning_rate))
  
  print(model.encoder.summary())
  print(model.decoder.summary())

  history = model.fit(x=train_images,
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