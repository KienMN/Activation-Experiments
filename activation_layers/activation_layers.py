from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras import backend as K

class DPReLU(Layer):
  def __init__(self,
               alpha_initializer='zeros',
               alpha_regularizer=None,
               alpha_constraint=None,
               beta_initializer='ones',
               beta_regularizer=None,
               beta_constraint=None,
               threshold_initializer='zeros',
               threshold_regularizer=None,
               threshold_constraint=None,
               bias_initializer='zeros',
               bias_regularizer=None,
               bias_constraint=None,
               shared_axes=None, **kwargs):
    super(DPReLU, self).__init__(**kwargs)

    self.supports_masking = True
    self.alpha_initializer = initializers.get(alpha_initializer)
    self.alpha_regularizer = regularizers.get(alpha_regularizer)
    self.alpha_constraint = constraints.get(alpha_constraint)

    self.beta_initializer = initializers.get(beta_initializer)
    self.beta_regularizer = regularizers.get(beta_regularizer)
    self.beta_constraint = constraints.get(beta_constraint)

    self.threshold_initializer = initializers.get(threshold_initializer)
    self.threshold_regularizer = regularizers.get(threshold_regularizer)
    self.threshold_constraint = constraints.get(threshold_constraint)

    self.bias_initializer = initializers.get(bias_initializer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.bias_constraint = constraints.get(bias_constraint)

    if shared_axes is None:
      self.shared_axes = None
    elif not isinstance(shared_axes, (list, tuple)):
      self.shared_axes = [shared_axes]
    else:
      self.shared_axes = list(shared_axes)

  def build(self, input_shape):
    param_shape = list(input_shape[1:])
    if self.shared_axes is not None:
      for i in self.shared_axes:
        param_shape[i - 1] = 1
    
    self.alpha = self.add_weight(
      shape=param_shape,
      name='alpha',
      initializer=self.alpha_initializer,
      regularizer=self.alpha_regularizer,
      constraint=self.alpha_constraint)

    self.threshold = self.add_weight(
      shape=param_shape,
      name='threshold',
      initializer=self.threshold_initializer,
      regularizer=self.threshold_regularizer,
      constraint=self.threshold_constraint)

    self.beta = self.add_weight(
      shape=param_shape,
      name='beta',
      initializer=self.beta_initializer,
      regularizer=self.beta_regularizer,
      constraint=self.beta_constraint)

    self.bias = self.add_weight(
      shape=param_shape,
      name='bias',
      initializer=self.bias_initializer,
      regularizer=self.bias_regularizer,
      constraint=self.bias_constraint)

    # Set input spec
    axes = {}
    if self.shared_axes:
      for i in range(1, len(input_shape)):
        if i not in self.shared_axes:
          axes[i] = input_shape[i]
    self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
    self.built = True

  def call(self, inputs):
    neg = -self.alpha * K.relu(-inputs + self.threshold)
    pos = self.beta * K.relu(inputs - self.threshold)
    return pos + neg + self.bias

  def get_config(self):
    config = {
      'alpha_initializer': initializers.serialize(self.alpha_initializer),
      'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
      'alpha_constraint': constraints.serialize(self.alpha_constraint),
      'threshold_initializer': initializers.serialize(self.threshold_initializer),
      'threshold_regularizer': regularizers.serialize(self.threshold_regularizer),
      'threshold_constraint': constraints.serialize(self.threshold_constraint),
      'beta_initializer': initializers.serialize(self.beta_initializer),
      'beta_regularizer': regularizers.serialize(self.beta_regularizer),
      'beta_constraint': constraints.serialize(self.beta_constraint),
      'bias_initializer': initializers.serialize(self.bias_initializer),
      'bias_regularizer': regularizers.serialize(self.bias_regularizer),
      'bias_constraint': constraints.serialize(self.bias_constraint),
      'shared_axes': self.shared_axes
    }
    base_config = super(DPReLU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape