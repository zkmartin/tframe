from __future__ import absolute_import
from __future__ import division
from __future__ import division

import numpy as np
import tensorflow as tf

from tframe import pedia
from tframe import with_graph

from tframe import Predictor
from tframe.nets.net import Net
from tframe.models import Feedforward
from tframe.layers import Linear
from tframe import initializers

from tframe import console
from tframe import losses
from tframe import metrics
from tframe import TFData

from tframe import FLAGS


class Bamboo(Predictor):
  def __init__(self, mark=None, **kwargs):
    # Call parent's initializer
    Predictor.__init__(self, mark)
    # Private fields
    self._output_list = []
    self._losses = []
    self._metrics = []
    self._train_ops = []
    self._var_list = []
    self._branch_index = 0
    self._identity_initial = kwargs.get('identity', False)


  def set_branch_index(self, index):
    # Sanity check
    if not 0 <= index < len(self._losses):
      raise IndexError('!! branch index should be between {} and {}'.format(
        0, len(self._losses)))

    self._branch_index = index
    self._loss = self._losses[index]
    self._metric = self._metrics[index]
    self._train_step = self._train_ops[index]
    self.outputs = self._output_list[index]

  @property
  def trunk_net(self):
      trunk_net = []
      for i, net in enumerate(self.children):
        if net.is_branch or i == -1:
          continue
        trunk_net.append(net)
      return trunk_net

  @with_graph
  def build(self, loss='cross_entropy', lr_list=None, optimizer=None,
            metric=None, metric_name='Metric'):
    if self._identity_initial:
      self._initial_define()
    Feedforward.build(self)
    # Check branch shapes
    output_shape = self.outputs.get_shape().as_list()
    for b_out in self.branch_outputs:
      assert isinstance(b_out, tf.Tensor)
      if b_out.get_shape().as_list() != output_shape:
        raise ValueError('!! Branch outputs in bamboo should have the same'
                         ' shape as the trunk output')
    # Initiate targets and add it to collection
    self._targets = tf.placeholder(self.outputs.dtype, output_shape,
                                   name='targets')
    tf.add_to_collection(pedia.default_feed_dict, self._targets)

    # Generate output list
    output_list = self.branch_outputs + [self.outputs]

    # Define losses
    loss_function = losses.get(loss)
    with tf.name_scope('Loss'):
      # Add branch outputs
      for output in output_list:
        self._losses.append(loss_function(self._targets, output))

    # Define metrics
    metric_function = metrics.get(metric)
    if metric_function is not None:
      pedia.memo[pedia.metric_name] = metric_name
      with tf.name_scope('Metric'):
        for output in output_list:
          self._metrics.append(metric_function(self._targets, output))

    # Define train step
    self._define_train_step(optimizer)

    # Sanity check
    assert len(self._losses) == len(self._metrics) == len(
      self.branch_outputs) + 1

    # Print status and model structure
    self.show_building_info(FeedforwardNet=self)

    # Launch session
    self.launch_model(FLAGS.overwrite and FLAGS.train)

    # Set built flag
    self._output_list = output_list
    self._built = True

  @with_graph
  def train(self, *args, branch_index=0, t_branch_s_index=0,  t_branch_e_index=0, lr_list=None, **kwargs):
    layer_train = kwargs.get('layer_train', False)
    if layer_train:
      return self._train(*args, branch_index=branch_index, **kwargs)
    else:
      return self._train_to_the_top(*args, branch_index_s=t_branch_s_index, branch_index_e=t_branch_e_index,
                                    lr_list=lr_list, **kwargs)

  @with_graph
  def _initial_define(self):
    for i, net in enumerate(self.children):
      if i == 0 or i == len(self.children) - 1:
        continue
      if net.is_branch is True:
        # define in traning step
        continue
      else:
          for f in net.children:
           if isinstance(f, Linear):
             f._weight_initializer = initializers.get('identity')
             f._bias_initializer = initializers.get(tf.zeros_initializer())

  def _initial_define_test(self):
    for net in self.trunk_net:
      for var in net.var_list:
        if 'biases' in var.name or 'bias' in var.name:
          var.initializers = initializers.get(tf.zeros_initializer())
        else:
          var.initializers = initializers.get('identity')

  @with_graph
  def _define_train_step(self, optimizer=None, var_list=None):
    assert len(self._losses) > 0
    with tf.name_scope('Optimizer'):
      if optimizer is None: optimizer = tf.train.AdamOptimizer(1e-4)
      self._optimizer = optimizer
      loss_index = 0
      var_list = []
      for i, net in enumerate(self.children):
        assert isinstance(net, Net)
        var_list += net.var_list
        if net.is_branch or i == len(self.children) - 1:
          self._train_ops.append(optimizer.minimize(
            loss=self._losses[loss_index], var_list=var_list))
          self._var_list.append(var_list)
          loss_index += 1
          var_list = []

    assert len(self._losses) == len(self._train_ops)

  @with_graph
  def _define_train_step_lr(self, lr_list=None):
    assert len(self._losses) > 0
    assert len(self.branches) + 1 == len(lr_list)
    with tf.name_scope('Optimizer'):
      var_list = []
      loss_index = 0
      lr_index = 0
      for i, net in enumerate(self.children):
        assert isinstance(net, Net)
        var_list += net.var_list
        if net.is_branch or i == len(self.children) - 1:
          optimizer = tf.train.AdamOptimizer(lr_list[lr_index])
          self._optimizer = optimizer
          self._train_ops.append(optimizer.minimize(
            loss=self._losses[loss_index], var_list=var_list))
          loss_index += 1
          lr_index += 1
          var_list = []

    assert len(self._losses) == len(self._train_ops)

  @with_graph
  def _train(self, *args, branch_index=0, **kwargs):
    self.set_branch_index(branch_index)
    # TODO
    freeze = kwargs.get('freeze', True)
    if not freeze: self._train_step = self._optimizer.minimize(self._loss)
    # Call parent's train method
    Predictor.train(self, *args, **kwargs)

  @with_graph
  def _train_to_the_top(self, *args, branch_index_s=0, branch_index_e=0, lr_list=None, **kwargs):
    if lr_list is None:lr_list = [0.000088] * (self.branches_num + 1)
    if branch_index_e == 0:
      train_end_index = self.branches_num + 1
    else:
      train_end_index = branch_index_e + 1
    for i in range(branch_index_s, train_end_index):
      self.set_branch_index(i)
      # TODO
      if i > 0:
         FLAGS.overwrite = False
         FLAGS.save_best = True
         self.launch_model(FLAGS.overwrite and FLAGS.train)
         if i == self.branches_num:
          self._branches_variables_assign(0, output=True)
         else:
          self._branches_variables_assign(i)
      self._optimizer_lr_modify(lr_list[i])
      self._train_step = self._optimizer.minimize(loss=self._losses[i], var_list=self._var_list[i])
      Predictor.train(self, *args, **kwargs)
      # if i > 0:
      #   self._optimizer_lr_modify(lr_list[i]*0.01)
      #   self._train_step = self._optimizer.minimize(loss=self._loss)
      #   Predictor.train(self, *args, **kwargs)
    FLAGS.overwrite = False
    self.launch_model(FLAGS.overwrite and FLAGS.train)
    self.set_branch_index(self.branches_num + 1)
    self._optimizer_lr_modify(lr_list[-1])
    self._train_step = self._optimizer.minimize(loss=self._loss)
    Predictor.train(self, *args, **kwargs)

  @with_graph
  def _variables_assign(self, index):
    value_weights = self.branches[index - 1].children[0]._weights
    ref_weights = self.branches[index].children[0]._weights
    value_bias = self.branches[index - 1].children[0]._biases
    ref_bias = self.branches[index].children[0]._biases
    tf.assign(ref_weights, value_weights)
    tf.assign(ref_bias, value_bias)

  @with_graph
  def _branches_variables_assign(self, index, output=False):
      for i in range(len(self.branches[index].var_list)):
        if output:
          self._session.run(tf.assign(self.children[-1].var_list[i], self.branches[-1].var_list[i]))
        else:
          self._session.run(tf.assign(self.branches[index].var_list[i], self.branches[index - 1].var_list[i]))

  def _optimizer_lr_modify(self, lr):
    if hasattr(self._optimizer, '_lr'):
      self._optimizer._lr = lr
    else:
      assert hasattr(self._optimizer, '_leanrning_rate')
      self._optimizer._learning_rate = lr

  def predict(self, data, **kwargs):
    index = kwargs.get('branch_index', 0)
    self.set_branch_index(index)
    # Call parent's predict method
    return Predictor.predict(self, data, **kwargs)


