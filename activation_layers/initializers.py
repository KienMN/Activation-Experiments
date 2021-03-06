import tensorflow as tf

def modified_he_normal(seed=None):
  return tf.keras.initializers.VarianceScaling(
    scale=2., mode="fan_avg", distribution="untruncated_normal", seed=seed
  )

def dprelu_normal(alpha=0.0, beta=1.0, seed=None):
  scale = 2.0 / (alpha ** 2 + beta ** 2)
  return tf.keras.initializers.VarianceScaling(
    scale=scale, mode="fan_avg", distribution="untruncated_normal", seed=seed
  )

def prelu_normal(alpha=0.0, seed=None):
  scale = 2.0 / (alpha ** 2 + 1)
  return tf.keras.initializers.VarianceScaling(
    scale=scale, mode="fan_avg", distribution="untruncated_normal", seed=seed
  )

def xavier_untruncated_normal(seed=None):
  return tf.keras.initializers.VarianceScaling(
    scale=1.0, mode="fan_avg", distribution="untruncated_normal", seed=seed
  )