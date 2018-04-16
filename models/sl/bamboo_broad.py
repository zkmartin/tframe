from __future__ import absolute_import
from __future__ import division
from __future__ import division

import numpy as np
import tensorflow as tf

from tframe import pedia
from tframe import with_graph

from tframe import Predictor
from tframe.nets.net import Net
from tframe.models.model import Model
from tframe.models import Feedforward
from tframe.layers import Linear
from tframe import initializers

from tframe import console
from tframe import losses
from tframe import metrics
from tframe import TFData

from tframe import FLAGS

class Bamboo_Broad(Model, Net):
    def __init__(self, mark=None, **kwargs):
        # Call parent's initializer
        Model.__init__(self, mark)
        Net.__init__(self, 'Bamboo_Broad_Net', inter_type=pedia.fork)
        assert self._inter_type == pedia.fork
        self.outputs = None
        # Private fields
        self._losses = []
        self._metrics = []
        self._train_ops = []
        self._var_list = []
        self._output_list = []
        self._branch_index = 0
        self._identity_initial = kwargs.get('ientity', False)

    def set_branch_index(self, index):
        self._loss = self._losses[index]
        self._train_step = self._train_ops[index]
        self._metric = self._metrics[index]
        self.outputs = self._output_list[index]

    @with_graph
    def build(self, loss='cross_entropy', optimizer=None,
              metric=None, metric_name='Metric'):
        if self._identity_initial:
            self._identity_define()

        self.outputs = self()

        # Initiate targets and add it to collection
        self._targets = tf.placeholder(name='targets',
                                       shape=self.branch_outputs[0].get_shape(),
                                       dtype=self.branch_outputs[0].dtype)
        tf.add_to_collection(pedia.default_feed_dict, self._targets)

        output_list = []
        for i in range(len(self.branch_outputs)):
            output_list.append(tf.add_n(self.branch_outputs[:(i + 1)]))

        # Define loss
        loss_function = losses.get(loss)
        with tf.name_scope('Loss'):
            for output in output_list:
                self._losses.append(loss_function(self._targets, output))

        # Define metrics
        metric_function = metrics.get(metric)
        if metric_function is not None:
            pedia.memo[pedia.metric_name] = metric_name
            with tf.name_scope('Metric'):
                for output in output_list:
                    self._metrics.append(metric_function(self._targets, output))

        # Define_train_step
        self._define_train_step(optimizer)

        # Sanity check
        assert len(self._losses) == len(self._metrics) == len(self.branch_outputs)

        # Print status and model structure
        self.show_building_info(Feedforward=self)

        # Launch session
        self.launch_model(FLAGS.overwrite and FLAGS.train)

        # Set built flag
        self._output_list = output_list
        self._built = True
        
    @with_graph
    def _identity_define(self):
        for i, net in enumerate(self.children):
            if i == 0:
                continue
            else:
                for var in net.var_list:
                    var.initializers = initializers.get(tf.zeros_initializer())

    @with_graph
    def _define_train_step(self, optimizer=None, var_list=None):
        with tf.name_scope('Optimizer'):
            if optimizer is None: optimizer = tf.train.AdamOptimizer(1e-4)
            self._optimizer = optimizer
            var_list = []
            loss_index = 0
            for i, net in enumerate(self.children):
                assert isinstance(net, Net)
                var_list.append(net.var_list)
                if net.is_branch:
                    self._var_list.append(var_list)
                    self._train_ops.append(self._optimizer.minimize(
                        loss=self._losses[loss_index], var_list=var_list))
                    loss_index += 1
                    var_list = []

    @with_graph
    def train(self, *args, branch_index=0, lr_list=None, **kwargs):
        if lr_list is None: lr_list = [0.000088]*self.branches_num
        self.set_branch_index(branch_index)
        freeze = kwargs.get('freeze', True)
        if not freeze:
            train_step = []
            for i in range(branch_index + 1):
                self._optimizer_lr_modify(lr_list[i])
                train_step.append(self._optimizer.minimize(loss=self._loss, var_list=self._var_list[i]))
            self._train_step = train_step

        Model.train(self, *args, **kwargs)

    def advanced_one_step(self, *args, lr_list=None, **kwargs):
        if lr_list is None: lr_list = [0.000088]*self.branches_num
        for i in range(self.branches_num):
            self.set_branch_index(i)
            if i > 0:
                FLAGS.overwrite = False
                FLAGS.save_best = True
            self._optimizer_lr_modify(lr_list[i])
            Model.train(self, *args, **kwargs)
        lr_list = [0.000088, 0.00088, 0.000088]
        self.train(branch_index=self.branches_num, lr_list=lr_list, **kwargs)

    def _optimizer_lr_modify(self, lr):
        if hasattr(self._optimizer, '_lr'):
            self._optimizer._lr = lr
        else:
            assert hasattr(self._optimizer, '_leanrning_rate')
            self._optimizer._learning_rate = lr

    def predict(self, data, **kwargs):
        index = kwargs.get('branch_index', 0)
        self.set_branch_index(index)

        # Sanity check
        if not isinstance(data, TFData):
            raise TypeError('!! Input data must be an instance of TFData')
        if not self.built: raise ValueError('!! Model not built yet')
        if self._session is None:
            self.launch_model(overwrite=False)

        if data.targets is None:
            outputs = self._session.run(
                self.outputs,
                feed_dict=self._get_default_feed_dict(data, is_training=False))
            return outputs
        else:
            outputs, loss = self._session.run(
                [self.outputs, self._loss],
                feed_dict=self._get_default_feed_dict(data, is_training=False))
            return outputs, loss





