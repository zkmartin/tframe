import tensorflow as tf

import tframe as tfr


def with_graph(meth):
  def wrapper(*args, **kwargs):
    obj = args[0]
    # For Model objects
    graph = obj.__dict__.get('_graph', None)
    # Nest methods with graph
    if graph is None: return meth(*args, **kwargs)
    else:
      assert isinstance(graph, tf.Graph)
      with graph.as_default(): return meth(*args, **kwargs)
  return wrapper


def init_with_graph(init):
  def wrapper(*args, **kwargs):
    assert isinstance(tfr.current_graph, tf.Graph)
    with tfr.current_graph.as_default(): init(*args, **kwargs)
  return wrapper


def single_input(_link):
  from tframe.layers.layer import Layer
  from tframe.nets.net import Net

  def wrapper(*args):
    assert isinstance(args[0], (Layer, Net))
    input_ = args[1]
    if isinstance(input_, list):
      if len(input_) != 1:
        raise ValueError('!! This function only accept single input')
      input_ = input_[0]
    if not isinstance(input_, tf.Tensor):
      raise TypeError('!! This layer only accept a Tensor as input')
    args = (args[0], input_) + args[2:]
    return _link(*args)

  return wrapper
