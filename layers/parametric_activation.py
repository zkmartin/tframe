from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.layers.layer import Layer
from tframe.layers.layer import single_input

from tframe import initializers


class Polynomial(Layer):
  """Polynomial activation function. ref: Vasilis Z. 1997"""
  is_nucleus = True

  full_name = 'polynomial'
  abbreviation = 'poly'

  def __init__(self, order, initializer=None):
    # Check input
    if order < 0: raise TypeError('!! order must be a non-negative integer')

    self.order = order
    self.coefs = None
    if initializer is None:
      self._initializer = tf.random_normal_initializer(stddev=0.02)
    tail = '{}'.format(order)
    self.full_name += tail


  @single_input
  def _link(self, input_, **kwargs):
    assert isinstance(input_, tf.Tensor)
    if self.coefs is not None: tf.get_variable_scope().reuse_variables()
    # Get input dimension
    D = input_.get_shape().as_list()[1]
    self.neuron_scale = [D]
    # Get variable and calculate output
    order_list = []
    self.coefs = tf.get_variable(
      'coefs', shape=(self.order + 1, D), dtype=input_.dtype,
      initializer=self._initializer)
    x = tf.ones_like(input_)
    for order in range(self.order + 1):
      order_list.append(tf.multiply(self.coefs[order], x))
      if order != self.order:
        x = tf.multiply(x, input_, name='x_{}'.format(order + 1))

    return tf.add_n(order_list)

class Polynomial_Relu(Layer):

  is_nucleus = True

  full_name = 'polynomial_relu'
  abbreviation = 'poly_relu'

  def __init__(self, order, initializer=None):
    # check input
    if order < 0: raise TypeError('!! order must be a non-negtive integer')

    self.order = order
    self.coefs = None
    if initializer is None:
      self._initializer = tf.random_normal_initializer(stddev=0.02)
    tail = '{}'.format(order)
    self.full_name += tail

  @single_input
  def _link(self, input_, **kwargs):
    assert isinstance(input_, tf.Tensor)
    if self.coefs is not None: tf.get_variable_scope().reuse_variables()

    D = input_.get_shape().as_list()[1]
    self.neuron_scale = [D]

    order_list = []
    self.coefs = tf.get_variable(
      name='coefs', shape=(self.order, D), dtype=tf.float32,
      initializer=self._initializer
    )
    x = tf.ones_like(input_)
    for order in range(1, self.order + 1):
      order_list.append(tf.multiply(self.coefs[order - 1], x))
      if order != self.order:
        x = tf.multiply(x, input_, name='x_{}'.format(order))

    return tf.nn.relu(tf.add_n(order_list))











