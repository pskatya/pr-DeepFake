from typing import Optional, Sequence, Text, Type, Union

import haiku as hk
import jax
import jax.numpy as jnp

from models import types


class BottleneckBlock(hk.Module):
  def __init__(self,
               channels: int,
               stride: Union[int, Sequence[int]],
               use_projection: bool,
               normalize_fn: Optional[types.NormalizeFn] = None,
               name: Optional[Text] = None):
    super(BottleneckBlock, self).__init__(name=name)
    self._channels = channels
    self._stride = stride
    self._use_projection = use_projection
    self._normalize_fn = normalize_fn

    if self._use_projection:
      self._proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          with_bias=False,
          padding='SAME',
          name='shortcut_conv')

    self._conv_0 = hk.Conv2D(
        output_channels=channels // 4,
        kernel_shape=1,
        stride=1,
        with_bias=False,
        padding='SAME',
        name='conv_0')

    self._conv_1 = hk.Conv2D(
        output_channels=channels // 4,
        kernel_shape=3,
        stride=stride,
        with_bias=False,
        padding='SAME',
        name='conv_1')

    self._conv_2 = hk.Conv2D(
        output_channels=channels,
        kernel_shape=1,
        stride=1,
        with_bias=False,
        padding='SAME',
        name='conv_2')

  def __call__(self,
               inputs,
               is_training):
    net = inputs
    shortcut = inputs

    for i, conv_i in enumerate([self._conv_0, self._conv_1, self._conv_2]):
      if self._normalize_fn is not None:
        net = self._normalize_fn(net, is_training=is_training)
      net = jax.nn.relu(net)
      if i == 0 and self._use_projection:
        shortcut = self._proj_conv(net)

      net = conv_i(net)

    return net + shortcut


class BasicBlock(hk.Module):
  def __init__(self,
               channels: int,
               stride: Union[int, Sequence[int]],
               use_projection: bool,
               normalize_fn: Optional[types.NormalizeFn] = None,
               name: Optional[Text] = None):
    super(BasicBlock, self).__init__(name=name)
    self._channels = channels
    self._stride = stride
    self._use_projection = use_projection
    self._normalize_fn = normalize_fn

    if self._use_projection:
      self._proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          with_bias=False,
          padding='SAME',
          name='shortcut_conv')

    self._conv_0 = hk.Conv2D(
        output_channels=channels,
        kernel_shape=1,
        stride=1,
        with_bias=False,
        padding='SAME',
        name='conv_0')

    self._conv_1 = hk.Conv2D(
        output_channels=channels,
        kernel_shape=3,
        stride=stride,
        with_bias=False,
        padding='SAME',
        name='conv_1')

  def __call__(self,
               inputs,
               is_training):
    net = inputs
    shortcut = inputs

    for i, conv_i in enumerate([self._conv_0, self._conv_1]):
      if self._normalize_fn is not None:
        net = self._normalize_fn(net, is_training=is_training)
      net = jax.nn.relu(net)
      if i == 0 and self._use_projection:
        shortcut = self._proj_conv(net)

      net = conv_i(net)

    return net + shortcut


class ResNetUnit(hk.Module):
  def __init__(self,
               channels: int,
               num_blocks: int,
               stride: Union[int, Sequence[int]],
               block_module: Type[BottleneckBlock],
               normalize_fn: Optional[types.NormalizeFn] = None,
               name: Optional[Text] = None,
               remat: bool = False):
    super(ResNetUnit, self).__init__(name=name)
    self._channels = channels
    self._num_blocks = num_blocks
    self._stride = stride
    self._normalize_fn = normalize_fn
    self._block_module = block_module
    self._remat = remat

  def __call__(self,
               inputs,
               is_training):

    input_channels = inputs.shape[-1]

    self._blocks = []
    for id_block in range(self._num_blocks):
      use_projection = id_block == 0 and self._channels != input_channels
      self._blocks.append(
          self._block_module(
              channels=self._channels,
              stride=self._stride if id_block == 0 else 1,
              use_projection=use_projection,
              normalize_fn=self._normalize_fn,
              name='block_%d' % id_block))

    net = inputs
    for block in self._blocks:
      if self._remat:
        net = hk.remat(lambda x: block(x, is_training=is_training))(net)  
      else:
        net = block(net, is_training=is_training)
    return net


class ResNetV2(hk.Module):
  VALID_ENDPOINTS = (
      'resnet_stem',
      'resnet_unit_0',
      'resnet_unit_1',
      'resnet_unit_2',
      'resnet_unit_3',
      'last_conv',
      'output',
  )

  def __init__(self,
               depth=50,
               num_classes: Optional[int] = 1000,
               width_mult: int = 1,
               normalize_fn: Optional[types.NormalizeFn] = None,
               name: Optional[Text] = None,
               remat: bool = False):

    super(ResNetV2, self).__init__(name=name)
    self._normalize_fn = normalize_fn
    self._num_classes = num_classes
    self._width_mult = width_mult

    self._strides = [1, 2, 2, 2]
    num_blocks = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3],
    }
    if depth not in num_blocks:
      raise ValueError(
          f'`depth` should be in {list(num_blocks.keys())} ({depth} given).')
    self._num_blocks = num_blocks[depth]

    if depth >= 50:
      self._block_module = BottleneckBlock
      self._channels = [256, 512, 1024, 2048]
    else:
      self._block_module = BasicBlock
      self._channels = [64, 128, 256, 512]

    self._initial_conv = hk.Conv2D(
        output_channels=64 * self._width_mult,
        kernel_shape=7,
        stride=2,
        with_bias=False,
        padding='SAME',
        name='initial_conv')

    if remat:
      self._initial_conv = hk.remat(self._initial_conv)

    self._block_groups = []
    for i in range(4):
      self._block_groups.append(
          ResNetUnit(
              channels=self._channels[i] * self._width_mult,
              num_blocks=self._num_blocks[i],
              block_module=self._block_module,
              stride=self._strides[i],
              normalize_fn=self._normalize_fn,
              name='block_group_%d' % i,
              remat=remat))

    if num_classes is not None:
      self._logits_layer = hk.Linear(
          output_size=num_classes, w_init=jnp.zeros, name='logits')

  def __call__(self, inputs, is_training, final_endpoint='output'):
    self._final_endpoint = final_endpoint
    net = self._initial_conv(inputs)
    net = hk.max_pool(
        net, window_shape=(1, 3, 3, 1),
        strides=(1, 2, 2, 1),
        padding='SAME')
    end_point = 'resnet_stem'
    if self._final_endpoint == end_point:
      return net

    for i_group, block_group in enumerate(self._block_groups):
      net = block_group(net, is_training=is_training)
      end_point = f'resnet_unit_{i_group}'
      if self._final_endpoint == end_point:
        return net

    end_point = 'last_conv'
    if self._final_endpoint == end_point:
      return net

    if self._normalize_fn is not None:
      net = self._normalize_fn(net, is_training=is_training)
      net = jax.nn.relu(net)
    net = jnp.mean(net, axis=[1, 2])

    assert self._final_endpoint == 'output'
    if self._num_classes is None:
      return net

    return self._logits_layer(net)
