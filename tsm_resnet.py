from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp

from models import tsm_utils as tsmu
from models import types


class TSMResNetBlock(hk.Module):
  def __init__(self,
               output_channels: int,
               stride: int,
               use_projection: bool,
               tsm_mode: str,
               normalize_fn: Optional[types.NormalizeFn] = None,
               channel_shift_fraction: float = 0.125,
               num_frames: int = 8,
               name: str = 'TSMResNetBlock'):
    super().__init__(name=name)
    self._output_channels = output_channels
    self._bottleneck_channels = output_channels // 4
    self._stride = stride
    self._use_projection = use_projection
    self._normalize_fn = normalize_fn
    self._tsm_mode = tsm_mode
    self._channel_shift_fraction = channel_shift_fraction
    self._num_frames = num_frames

  def __call__(self,
               inputs: types.TensorLike,
               is_training: bool = True) -> jnp.ndarray:
    preact = inputs
    if self._normalize_fn is not None:
      preact = self._normalize_fn(preact, is_training=is_training)
    preact = jax.nn.relu(preact)

    if self._use_projection:
      shortcut = hk.Conv2D(
          output_channels=self._output_channels,
          kernel_shape=1,
          stride=self._stride,
          with_bias=False,
          padding='SAME',
          name='shortcut_conv')(
              preact)
    else:
      shortcut = inputs
    if self._channel_shift_fraction != 0:
      preact = tsmu.apply_temporal_shift(
          preact, tsm_mode=self._tsm_mode, num_frames=self._num_frames,
          channel_shift_fraction=self._channel_shift_fraction)

    # First convolution.
    residual = hk.Conv2D(
        self._bottleneck_channels,
        kernel_shape=1,
        stride=1,
        with_bias=False,
        padding='SAME',
        name='conv_0')(
            preact)

    # Second convolution.
    if self._normalize_fn is not None:
      residual = self._normalize_fn(residual, is_training=is_training)
    residual = jax.nn.relu(residual)
    residual = hk.Conv2D(
        output_channels=self._bottleneck_channels,
        kernel_shape=3,
        stride=self._stride,
        with_bias=False,
        padding='SAME',
        name='conv_1')(
            residual)

    # Third convolution.
    if self._normalize_fn is not None:
      residual = self._normalize_fn(residual, is_training=is_training)
    residual = jax.nn.relu(residual)
    residual = hk.Conv2D(
        output_channels=self._output_channels,
        kernel_shape=1,
        stride=1,
        with_bias=False,
        padding='SAME',
        name='conv_2')(
            residual)

    output = shortcut + residual
    return output


class TSMResNetUnit(hk.Module):
  def __init__(self,
               output_channels: int,
               num_blocks: int,
               stride: int,
               tsm_mode: str,
               num_frames: int,
               normalize_fn: Optional[types.NormalizeFn] = None,
               channel_shift_fraction: float = 0.125,
               name: str = 'tsm_resnet_unit'):
    super().__init__(name=name)
    self._output_channels = output_channels
    self._num_blocks = num_blocks
    self._normalize_fn = normalize_fn
    self._stride = stride
    self._tsm_mode = tsm_mode
    self._channel_shift_fraction = channel_shift_fraction
    self._num_frames = num_frames

  def __call__(self,
               inputs: types.TensorLike,
               is_training: bool):
    net = inputs
    for idx_block in range(self._num_blocks):
      net = TSMResNetBlock(
          self._output_channels,
          stride=self._stride if idx_block == 0 else 1,
          use_projection=idx_block == 0,
          normalize_fn=self._normalize_fn,
          tsm_mode=self._tsm_mode,
          channel_shift_fraction=self._channel_shift_fraction,
          num_frames=self._num_frames,
          name=f'block_{idx_block}')(
              net, is_training=is_training)
    return net


class TSMResNetV2(hk.Module):
  VALID_ENDPOINTS = (
      'tsm_resnet_stem',
      'tsm_resnet_unit_0',
      'tsm_resnet_unit_1',
      'tsm_resnet_unit_2',
      'tsm_resnet_unit_3',
      'last_conv',
      'Embeddings',
  )

  def __init__(self,
               normalize_fn: Optional[types.NormalizeFn] = None,
               depth: int = 50,
               num_frames: int = 16,
               channel_shift_fraction: float = 0.125,
               width_mult: int = 1,
               name: str = 'TSMResNetV2'):
    super().__init__(name=name)

    if not 0. <= channel_shift_fraction <= 1.0:
      raise ValueError(
          f'channel_shift_fraction ({channel_shift_fraction})'
          ' has to be in [0, 1].')

    self._num_frames = num_frames

    self._channels = (256, 512, 1024, 2048)
    self._strides = (1, 2, 2, 2)

    num_blocks = {
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3),
        200: (3, 24, 36, 3),
    }
    if depth not in num_blocks:
      raise ValueError(
          f'`depth` should be in {list(num_blocks.keys())} ({depth} given).')
    self._num_blocks = num_blocks[depth]

    self._width_mult = width_mult
    self._channel_shift_fraction = channel_shift_fraction
    self._normalize_fn = normalize_fn

  def __call__(
      self,
      inputs: types.TensorLike,
      is_training: bool = True,
      final_endpoint: str = 'Embeddings'):

    inputs, tsm_mode, num_frames = tsmu.prepare_inputs(inputs)
    num_frames = num_frames or self._num_frames

    self._final_endpoint = final_endpoint
    if self._final_endpoint not in self.VALID_ENDPOINTS:
      raise ValueError(f'Unknown final endpoint {self._final_endpoint}')

    end_point = 'tsm_resnet_stem'
    net = hk.Conv2D(
        output_channels=64 * self._width_mult,
        kernel_shape=7,
        stride=2,
        with_bias=False,
        name=end_point,
        padding='SAME')(
            inputs)
    net = hk.MaxPool(
        window_shape=(1, 3, 3, 1),
        strides=(1, 2, 2, 1),
        padding='SAME')(
            net)
    if self._final_endpoint == end_point:
      return net
    
    for unit_id, (channels, num_blocks, stride) in enumerate(
        zip(self._channels, self._num_blocks, self._strides)):
      end_point = f'tsm_resnet_unit_{unit_id}'
      net = TSMResNetUnit(
          output_channels=channels * self._width_mult,
          num_blocks=num_blocks,
          stride=stride,
          normalize_fn=self._normalize_fn,
          channel_shift_fraction=self._channel_shift_fraction,
          num_frames=num_frames,
          tsm_mode=tsm_mode,
          name=end_point)(
              net, is_training=is_training)
      if self._final_endpoint == end_point:
        return net

    if self._normalize_fn is not None:
      net = self._normalize_fn(net, is_training=is_training)
    net = jax.nn.relu(net)

    end_point = 'last_conv'
    if self._final_endpoint == end_point:
      return net
    net = jnp.mean(net, axis=(1, 2))
    net = tsmu.prepare_outputs(net, tsm_mode, num_frames)
    assert self._final_endpoint == 'Embeddings'
    return net
