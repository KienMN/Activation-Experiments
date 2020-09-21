import tensorflow as tf

def modified_he_normal(seed=None):
  return tf.keras.initializers.VarianceScaling(
    scale=2., mode="fan_avg", distribution="truncated_normal", seed=seed
  )

def dprelu_normal(alpha=0.0, beta=1.0, seed=None):
  scale = 2 * (alpha ** 2 + beta ** 2)
  return tf.keras.initializers.VarianceScaling(
    scale=scale, mode="fan_avg", distribution="truncated_normal", seed=seed
  )