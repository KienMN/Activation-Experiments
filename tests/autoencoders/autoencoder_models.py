import tensorflow as tf
import numpy as np
from activation_layers import DPReLU, FReLU
from activation_layers import modified_he_normal, dprelu_normal, prelu_normal, xavier_untruncated_normal

class BaseAutoEncoder(tf.keras.Model):
  def __init__(self, *args, **kwargs):
    # self.input_dims = input_dims
    # self.latent_dim = latent_dim
    super(BaseAutoEncoder, self).__init__()
    self.encoder = None
    self.decoder = None

  def call(self, x, apply_sigmoid=True, training=False):
    if self.encoder is None or self.decoder is None:
      raise Exception('Encoder of Decoder is not implemented.')
    latent_code = self.encoder(x, training=training)
    reconstructed_input = self.decoder(latent_code, training=training)
    if apply_sigmoid:
      probs = tf.nn.sigmoid(reconstructed_input)
      return probs
    return reconstructed_input

class AutoEncoderWithBatchNormReLU(BaseAutoEncoder):
  def  __init__(self, input_dims, latent_dim):
    super(AutoEncoderWithBatchNormReLU, self).__init__()

    self.encoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=input_dims),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1000),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dense(500),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dense(250),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dense(latent_dim)
    ])

    self.decoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
      tf.keras.layers.Dense(250),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dense(500),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dense(1000),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dense(np.prod(input_dims)),
      tf.keras.layers.Reshape(input_dims)
    ])

class MnistAutoencoder(BaseAutoEncoder):
  def __init__(self,
              input_dims,
              latent_dim,
              activation='relu',
              alpha=0.,
              beta=1.,
              shared_axes=None,
              weight_initialization='xavier'):

    super(MnistAutoencoder, self).__init__()

    # Initial value of alpha and beta
    if alpha is not None:
      alpha_initializer = tf.keras.initializers.Constant(value=alpha)
    else:
      alpha_initializer = 'zeros'

    if beta is not None:
      beta_initializer = tf.keras.initializers.Constant(value=beta)
    else:
      beta_initializer = 'ones'

    # Weight initializer
    if weight_initialization == 'xavier':
      initializer = tf.keras.initializers.GlorotNormal()
    elif weight_initialization == 'xavier_untruncated':
      initializer = xavier_untruncated_normal()
    elif weight_initialization == 'he':
      initializer = tf.keras.initializers.he_normal()
    elif weight_initialization == 'modified_he':
      initializer = modified_he_normal()
    elif (weight_initialization == 'dprelu') and (alpha is not None) and (beta is not None):
      initializer = dprelu_normal(alpha=alpha, beta=beta)
    elif (weight_initialization == 'prelu') and (alpha is not None):
      initializer = prelu_normal(alpha=alpha)
    else:
      raise ValueError('No valid weight initializer')

    self.encoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=input_dims),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1000,
                            kernel_initializer=initializer),
      activation_layers(activation=activation,
                        alpha_initializer=alpha_initializer,
                        beta_initializer=beta_initializer,
                        shared_axes=shared_axes),
      tf.keras.layers.Dense(500,
                            kernel_initializer=initializer),
      activation_layers(activation=activation,
                        alpha_initializer=alpha_initializer,
                        beta_initializer=beta_initializer,
                        shared_axes=shared_axes),
      tf.keras.layers.Dense(250,
                            kernel_initializer=initializer),
      activation_layers(activation=activation,
                        alpha_initializer=alpha_initializer,
                        beta_initializer=beta_initializer,
                        shared_axes=shared_axes),
      tf.keras.layers.Dense(latent_dim,
                            kernel_initializer=initializer)
    ])

    self.decoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
      tf.keras.layers.Dense(250,
                            kernel_initializer=initializer),
      activation_layers(activation=activation,
                        alpha_initializer=alpha_initializer,
                        beta_initializer=beta_initializer,
                        shared_axes=shared_axes),
      tf.keras.layers.Dense(500,
                            kernel_initializer=initializer),
      activation_layers(activation=activation,
                        alpha_initializer=alpha_initializer,
                        beta_initializer=beta_initializer,
                        shared_axes=shared_axes),
      tf.keras.layers.Dense(1000,
                            kernel_initializer=initializer),
      activation_layers(activation=activation,
                        alpha_initializer=alpha_initializer,
                        beta_initializer=beta_initializer,
                        shared_axes=shared_axes),
      tf.keras.layers.Dense(np.prod(input_dims),
                            kernel_initializer=initializer),
      tf.keras.layers.Reshape(input_dims)
    ])

class Cifar10ConvAutoencoder(BaseAutoEncoder):
  def __init__(self,
              input_dims,
              latent_dim,
              hidden_dim=1024,
              activation='relu',
              alpha=0.,
              beta=1.,
              weight_initialization='xavier'):
    super(Cifar10ConvAutoencoder, self).__init__()

    # Initial value of alpha and beta
    if alpha is not None:
      alpha_initializer = tf.keras.initializers.Constant(value=alpha)
    else:
      alpha = 0.
      alpha_initializer = 'zeros'

    if beta is not None:
      beta_initializer = tf.keras.initializers.Constant(value=beta)
    else:
      beta = 1.
      beta_initializer = 'ones'

    # Weight initializer
    if weight_initialization == 'xavier':
      initializer = tf.keras.initializers.GlorotNormal()
    elif weight_initialization == 'xavier_untruncated':
      initializer = xavier_untruncated_normal()
    elif weight_initialization == 'he':
      initializer = tf.keras.initializers.he_normal()
    elif weight_initialization == 'modified_he':
      initializer = modified_he_normal()
    elif (weight_initialization == 'dprelu') and (alpha is not None) and (beta is not None):
      initializer = dprelu_normal(alpha=alpha, beta=beta)
    elif (weight_initialization == 'prelu') and (alpha is not None):
      initializer = prelu_normal(alpha=alpha)
    else:
      raise ValueError('No valid weight initializer')

    self.encoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=input_dims),
      tf.keras.layers.Conv2D(64,
                            kernel_size=3,
                            strides=(2, 2),
                            kernel_initializer=initializer),
      activation_layers(activation=activation,
                        alpha_initializer=alpha_initializer,
                        beta_initializer=beta_initializer,
                        shared_axes=[1, 2]),
      tf.keras.layers.Conv2D(128,
                            kernel_size=3,
                            strides=(2, 2),
                            kernel_initializer=initializer),
      activation_layers(activation=activation,
                        alpha_initializer=alpha_initializer,
                        beta_initializer=beta_initializer,
                        shared_axes=[1, 2]),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(hidden_dim,
                            kernel_initializer=initializer),
      activation_layers(activation=activation,
                        alpha_initializer=alpha_initializer,
                        beta_initializer=beta_initializer,
                        shared_axes=None),
      # No activation
      tf.keras.layers.Dense(latent_dim,
                            kernel_initializer=initializer)
    ])

    self.decoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(latent_dim)),
      tf.keras.layers.Dense(hidden_dim,
                            kernel_initializer=initializer),
      activation_layers(activation=activation,
                        alpha_initializer=alpha_initializer,
                        beta_initializer=beta_initializer,
                        shared_axes=None),
      tf.keras.layers.Dense(8*8*128,
                            kernel_initializer=initializer),
      tf.keras.layers.Reshape(target_shape=(8, 8, 128)),
      activation_layers(activation=activation,
                        alpha_initializer=alpha_initializer,
                        beta_initializer=beta_initializer,
                        shared_axes=[1, 2]),
      tf.keras.layers.Conv2DTranspose(filters=64,
                                      kernel_size=3,
                                      strides=(2, 2),
                                      padding='SAME',
                                      kernel_initializer=initializer),
      activation_layers(activation=activation,
                        alpha_initializer=alpha_initializer,
                        beta_initializer=beta_initializer,
                        shared_axes=[1, 2]),

      # No activation
      tf.keras.layers.Conv2DTranspose(filters=input_dims[-1],
                                      kernel_size=3,
                                      strides=(2, 2),
                                      padding='SAME',
                                      kernel_initializer=initializer)
    ])

def activation_layers(activation='relu',
                      alpha_initializer='zeros',
                      beta_initializer='ones',
                      shared_axes=None):
  if activation == 'relu':
    return tf.keras.layers.ReLU()
  elif activation == 'elu':
    return tf.keras.layers.ELU()
  elif activation == 'prelu':
    return tf.keras.layers.PReLU(alpha_initializer=alpha_initializer,
                                shared_axes=shared_axes)
  elif activation == 'frelu':
    return FReLU(shared_axes=shared_axes)
  elif activation == 'dprelu':
    return DPReLU(alpha_initializer=alpha_initializer,
                  beta_initializer=beta_initializer,
                  shared_axes=shared_axes)