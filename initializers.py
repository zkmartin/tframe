from __future__ import  absolute_import
import tensorflow as tf
import numpy as np

import six

from tensorflow.python.ops import init_ops


def glorot_uniform():
  return init_ops.glorot_uniform_initializer()

def identity_initializer(value):
  return _identity_initializer(value)

def _identity_initializer(value):
  def _initialier(shape, dtype=tf.float32, partition_info=None):
    input_n = value
    if value > shape[1]: raise ValueError('for identify initializer:the input_shape width must be less than the current shape!!!')
    initial_array_identify = np.zeros((input_n, input_n), dtype=float)
    initial_array_zeros = np.zeros((shape[0], (shape[1] - input_n)), dtype=float)
    for i in range(input_n):
      initial_array_identify[i, i] = 1
    initial_array = np.hstack((initial_array_identify, initial_array_zeros))
    return tf.constant(initial_array, dtype=dtype)
  return _initialier

def get(identifier):
  if identifier is None or isinstance(identifier, init_ops.Initializer):
    return identifier
  elif isinstance(identifier, six.string_types):
    # If identifier is a string
    identifier = identifier.lower()
    if identifier in ['glorot_uniform', 'xavier_uniform']:
      return glorot_uniform()
    elif identifier in ['identify', 'identify_initial']:
      return identity_initializer
    else:
      # Find initializer in tensorflow.python.ops.init_ops
      initializer = (
        init_ops.__dict__.get(identifier, None) or
        init_ops.__dict__.get('{}_initializer'.format(identifier),None))
      # If nothing is found
      if initializer is None:
        raise ValueError('Can not resolve "{}"'.format(identifier))
      # Return initializer with default parameters
      return initializer
  else:
    raise TypeError('identifier must be a Initializer or a string')


if __name__ == '__main__':
  print(get('glorot_normal'))
