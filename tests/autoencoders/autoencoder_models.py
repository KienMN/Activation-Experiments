import tensorflow as tf
import numpy as np
from activation_layers import DPReLU, FReLU

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

# class AutoEncoderWithReLU(BaseAutoEncoder):
#   def  __init__(self, input_dims, latent_dim):
#     super(AutoEncoderWithReLU, self).__init__()

#     self.encoder = tf.keras.Sequential([
#       tf.keras.layers.InputLayer(input_shape=input_dims),
#       tf.keras.layers.Flatten(),
#       tf.keras.layers.Dense(1000, activation='relu'),
#       tf.keras.layers.Dense(500, activation='relu'),
#       tf.keras.layers.Dense(250, activation='relu'),
#       tf.keras.layers.Dense(latent_dim)
#     ])

#     self.decoder = tf.keras.Sequential([
#       tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
#       tf.keras.layers.Dense(250, activation='relu'),
#       tf.keras.layers.Dense(500, activation='relu'),
#       tf.keras.layers.Dense(1000, activation='relu'),
#       tf.keras.layers.Dense(np.prod(input_dims)),
#       tf.keras.layers.Reshape(input_dims)
#     ])

# class AutoEncoderWithELU(BaseAutoEncoder):
#   def  __init__(self, input_dims, latent_dim):
#     super(AutoEncoderWithELU, self).__init__()

#     self.encoder = tf.keras.Sequential([
#       tf.keras.layers.InputLayer(input_shape=input_dims),
#       tf.keras.layers.Flatten(),
#       tf.keras.layers.Dense(1000, activation='elu'),
#       tf.keras.layers.Dense(500, activation='elu'),
#       tf.keras.layers.Dense(250, activation='elu'),
#       tf.keras.layers.Dense(latent_dim)
#     ])

#     self.decoder = tf.keras.Sequential([
#       tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
#       tf.keras.layers.Dense(250, activation='elu'),
#       tf.keras.layers.Dense(500, activation='elu'),
#       tf.keras.layers.Dense(1000, activation='elu'),
#       tf.keras.layers.Dense(np.prod(input_dims)),
#       tf.keras.layers.Reshape(input_dims)
#     ])

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

# class AutoEncoderWithPReLU(BaseAutoEncoder):
#   def  __init__(self, input_dims, latent_dim, shared_axes=None):
#     super(AutoEncoderWithPReLU, self).__init__()

#     self.encoder = tf.keras.Sequential([
#       tf.keras.layers.InputLayer(input_shape=input_dims),
#       tf.keras.layers.Flatten(),
#       tf.keras.layers.Dense(1000),
#       tf.keras.layers.PReLU(shared_axes=shared_axes),
#       tf.keras.layers.Dense(500),
#       tf.keras.layers.PReLU(shared_axes=shared_axes),
#       tf.keras.layers.Dense(250),
#       tf.keras.layers.PReLU(shared_axes=shared_axes),
#       tf.keras.layers.Dense(latent_dim)
#     ])

#     self.decoder = tf.keras.Sequential([
#       tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
#       tf.keras.layers.Dense(250),
#       tf.keras.layers.PReLU(shared_axes=shared_axes),
#       tf.keras.layers.Dense(500),
#       tf.keras.layers.PReLU(shared_axes=shared_axes),
#       tf.keras.layers.Dense(1000),
#       tf.keras.layers.PReLU(shared_axes=shared_axes),
#       tf.keras.layers.Dense(np.prod(input_dims)),
#       tf.keras.layers.Reshape(input_dims)
#     ])

# class AutoEncoderWithFReLU(BaseAutoEncoder):
#   def  __init__(self, input_dims, latent_dim, shared_axes=None):
#     super(AutoEncoderWithFReLU, self).__init__()

#     self.encoder = tf.keras.Sequential([
#       tf.keras.layers.InputLayer(input_shape=input_dims),
#       tf.keras.layers.Flatten(),
#       tf.keras.layers.Dense(1000),
#       FReLU(shared_axes=shared_axes),
#       tf.keras.layers.Dense(500),
#       FReLU(shared_axes=shared_axes),
#       tf.keras.layers.Dense(250),
#       FReLU(shared_axes=shared_axes),
#       tf.keras.layers.Dense(latent_dim)
#     ])

#     self.decoder = tf.keras.Sequential([
#       tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
#       tf.keras.layers.Dense(250),
#       FReLU(shared_axes=shared_axes),
#       tf.keras.layers.Dense(500),
#       FReLU(shared_axes=shared_axes),
#       tf.keras.layers.Dense(1000),
#       FReLU(shared_axes=shared_axes),
#       tf.keras.layers.Dense(np.prod(input_dims)),
#       tf.keras.layers.Reshape(input_dims)
#     ])

# class AutoEncoderWithDPReLU(BaseAutoEncoder):
#   def  __init__(self, input_dims, latent_dim, shared_axes=None):
#     super(AutoEncoderWithDPReLU, self).__init__()

#     self.encoder = tf.keras.Sequential([
#       tf.keras.layers.InputLayer(input_shape=input_dims),
#       tf.keras.layers.Flatten(),
#       tf.keras.layers.Dense(1000),
#       DPReLU(shared_axes=shared_axes),
#       tf.keras.layers.Dense(500),
#       DPReLU(shared_axes=shared_axes),
#       tf.keras.layers.Dense(250),
#       DPReLU(shared_axes=shared_axes),
#       tf.keras.layers.Dense(latent_dim)
#     ])

#     self.decoder = tf.keras.Sequential([
#       tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
#       tf.keras.layers.Dense(250),
#       DPReLU(shared_axes=shared_axes),
#       tf.keras.layers.Dense(500),
#       DPReLU(shared_axes=shared_axes),
#       tf.keras.layers.Dense(1000),
#       DPReLU(shared_axes=shared_axes),
#       tf.keras.layers.Dense(np.prod(input_dims)),
#       tf.keras.layers.Reshape(input_dims)
#     ])

class MnistAutoencoder(BaseAutoEncoder):
  def __init__(self, input_dims, latent_dim, activation='relu', shared_axes=None):
    super(MnistAutoencoder, self).__init__()

    self.encoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=input_dims),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1000),
      activation_layers(activation=activation, shared_axes=shared_axes),
      tf.keras.layers.Dense(500),
      activation_layers(activation=activation, shared_axes=shared_axes),
      tf.keras.layers.Dense(250),
      activation_layers(activation=activation, shared_axes=shared_axes),
      tf.keras.layers.Dense(latent_dim)
    ])

    self.decoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
      tf.keras.layers.Dense(250),
      activation_layers(activation=activation, shared_axes=shared_axes),
      tf.keras.layers.Dense(500),
      activation_layers(activation=activation, shared_axes=shared_axes),
      tf.keras.layers.Dense(1000),
      activation_layers(activation=activation, shared_axes=shared_axes),
      tf.keras.layers.Dense(np.prod(input_dims)),
      tf.keras.layers.Reshape(input_dims)
    ])

class Cifar10ConvAutoencoder(BaseAutoEncoder):
  def __init__(self, input_dims, latent_dim, hidden_dim=1024, activation='relu'):
    super(Cifar10ConvAutoencoder, self).__init__()

    self.encoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=input_dims),
      tf.keras.layers.Conv2D(64, kernel_size=3, strides=(2, 2)),
      activation_layers(activation=activation, shared_axes=[1, 2]),
      tf.keras.layers.Conv2D(128, kernel_size=3, strides=(2, 2)),
      activation_layers(activation=activation, shared_axes=[1, 2]),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(hidden_dim),
      activation_layers(activation=activation, shared_axes=None),
      # No activation
      tf.keras.layers.Dense(latent_dim)
    ])

    self.decoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(latent_dim)),
      tf.keras.layers.Dense(hidden_dim, activation=None),
      activation_layers(activation=activation, shared_axes=None),
      tf.keras.layers.Dense(8*8*128, activation=None),
      tf.keras.layers.Reshape(target_shape=(8, 8, 128)),
      activation_layers(activation=activation, shared_axes=[1, 2]),
      tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='SAME', activation=None),
      activation_layers(activation=activation, shared_axes=[1, 2]),

      # No activation
      tf.keras.layers.Conv2DTranspose(filters=input_dims[-1], kernel_size=3, strides=(2, 2), padding='SAME')
    ])

def activation_layers(activation='relu', shared_axes=None):
  if activation == 'relu':
    return tf.keras.layers.ReLU()
  elif activation == 'elu':
    return tf.keras.layers.ELU()
  elif activation == 'prelu':
    return tf.keras.layers.PReLU(shared_axes=shared_axes)
  elif activation == 'frelu':
    return FReLU(shared_axes=shared_axes)
  elif activation == 'dprelu':
    return DPReLU(shared_axes=shared_axes)