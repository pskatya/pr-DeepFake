import collections
from typing import Optional, Sequence

import haiku as hk
import jax
from jax import numpy as jnp

from models import types


class _MaxPool(hk.MaxPool):

  def __call__(self,
               x: types.TensorLike,
               is_training: bool = True) -> jnp.ndarray:
    del is_training  # Unused.
    return super().__call__(x)


def self_gating(inputs: types.TensorLike) -> jnp.ndarray:
  if inputs.ndim != 5:
    raise ValueError(
        f'Expected an input of shape `[B, T, H, W, C]` but got {inputs.shape}.')

  input_shape = inputs.shape
  num_channels = input_shape[4]
  spatiotemporal_average = jnp.mean(inputs, axis=(1, 2, 3))
  weights = hk.Linear(num_channels, name='self_gating')(spatiotemporal_average)
  weights = jax.nn.sigmoid(weights)
  return jnp.multiply(weights[:, None, None, None, :], inputs)


class SUnit3D(hk.Module):
  def __init__(
      self,
      output_channels: int,
      kernel_shape: Sequence[int] = (1, 1, 1),
      stride: Sequence[int] = (1, 1, 1),
      with_bias: bool = False,
      separable: bool = False,
      normalize_fn: Optional[types.NormalizeFn] = None,
      activation_fn: Optional[types.ActivationFn] = jax.nn.relu,
      self_gating_fn: Optional[types.GatingFn] = None,
      name='SUnit3D'):
    
    super().__init__(name=name)

    if len(kernel_shape) != 3:
      raise ValueError(
          'Given `kernel_shape` must have length 3 but has length '
          f'{len(kernel_shape)}.')
    if len(stride) != 3:
      raise ValueError(
          f'Given `stride` must have length 3 but has length {len(stride)}.')

    self._normalize_fn = normalize_fn
    self._activation_fn = activation_fn
    self._self_gating_fn = self_gating_fn

    k0, k1, k2 = kernel_shape
    if separable and k1 != 1:
      spatial_kernel_shape = [1, k1, k2]
      temporal_kernel_shape = [k0, 1, 1]
      s0, s1, s2 = stride
      spatial_stride = [1, s1, s2]
      temporal_stride = [s0, 1, 1]
      self._convolutions = [
          hk.Conv3D(
              output_channels=output_channels,
              kernel_shape=spatial_kernel_shape,
              stride=spatial_stride,
              padding='SAME',
              with_bias=with_bias),
          hk.Conv3D(
              output_channels=output_channels,
              kernel_shape=temporal_kernel_shape,
              stride=temporal_stride,
              padding='SAME',
              with_bias=with_bias)
      ]

    else:
      self._convolutions = [
          hk.Conv3D(
              output_channels=output_channels,
              kernel_shape=kernel_shape,
              stride=stride,
              padding='SAME',
              with_bias=with_bias)]

  def __call__(
      self,
      inputs: types.TensorLike,
      is_training: bool) -> jnp.ndarray:
    x = inputs
    for conv in self._convolutions:
      x = conv(x)
      if self._normalize_fn is not None:
        x = self._normalize_fn(x, is_training=is_training)
      if self._activation_fn is not None:
        x = self._activation_fn(x)
    if self._self_gating_fn:
      x = self._self_gating_fn(x)
    return x


class InceptionBlockV13D(hk.Module):

  def __init__(self,
               output_channels: Sequence[int],
               normalize_fn: Optional[types.NormalizeFn],
               temporal_kernel_size: int = 3,
               self_gating_fn: Optional[types.GatingFn] = None,
               name: str = 'InceptionBlockV13D'):
    super().__init__(name=name)

    if len(output_channels) != 6:
      raise ValueError(
          'Given `output_channels` must have length 6 but has length '
          f'{len(output_channels)}.')

    self._output_channels = output_channels
    self._normalize_fn = normalize_fn
    self._temporal_kernel_size = temporal_kernel_size

    if self_gating_fn is None:
      self._self_gating_fn = lambda x: x
    else:
      self._self_gating_fn = self_gating_fn

  def __call__(
      self,
      inputs: types.TensorLike,
      is_training: bool) -> jnp.ndarray:
    branch_0 = SUnit3D(
        output_channels=self._output_channels[0],
        kernel_shape=(1, 1, 1),
        separable=False,
        normalize_fn=self._normalize_fn,
        self_gating_fn=self._self_gating_fn,
        name='Branch_0_Conv2d_0a_1x1')(
            inputs, is_training=is_training)

    # Branch 1
    branch_1 = SUnit3D(
        output_channels=self._output_channels[1],
        kernel_shape=(1, 1, 1),
        separable=False,
        normalize_fn=self._normalize_fn,
        self_gating_fn=None,
        name='Branch_1_Conv2d_0a_1x1')(
            inputs, is_training=is_training)
    branch_1 = SUnit3D(
        output_channels=self._output_channels[2],
        kernel_shape=(self._temporal_kernel_size, 3, 3),
        separable=True,
        normalize_fn=self._normalize_fn,
        self_gating_fn=self._self_gating_fn,
        name='Branch_1_Conv2d_0b_3x3')(
            branch_1, is_training=is_training)

    # Branch 2
    branch_2 = SUnit3D(
        output_channels=self._output_channels[3],
        kernel_shape=(1, 1, 1),
        separable=False,
        normalize_fn=self._normalize_fn,
        self_gating_fn=None,
        name='Branch_2_Conv2d_0a_1x1')(
            inputs, is_training=is_training)
    branch_2 = SUnit3D(
        output_channels=self._output_channels[4],
        kernel_shape=(self._temporal_kernel_size, 3, 3),
        separable=True,
        normalize_fn=self._normalize_fn,
        self_gating_fn=self._self_gating_fn,
        name='Branch_2_Conv2d_0b_3x3')(
            branch_2, is_training=is_training)

    # Branch 3
    branch_3 = hk.MaxPool(
        window_shape=(1, 3, 3, 3, 1),
        strides=(1, 1, 1, 1, 1),
        padding='SAME',
        name='Branch_3_MaxPool_0a_3x3')(
            inputs)
    branch_3 = SUnit3D(
        output_channels=self._output_channels[5],
        kernel_shape=(1, 1, 1),
        separable=False,
        normalize_fn=self._normalize_fn,
        self_gating_fn=self._self_gating_fn,
        name='Branch_3_Conv2d_0b_1x1')(
            branch_3, is_training=is_training)

    return jnp.concatenate((branch_0, branch_1, branch_2, branch_3), axis=4)


_Layer = collections.namedtuple('_Layer', ('name', 'module', 'kwargs'))


class S3D(hk.Module):
  VALID_ENDPOINTS = (
      'Conv2d_1a_7x7',
      'MaxPool_2a_3x3',
      'Conv2d_2b_1x1',
      'Conv2d_2c_3x3',
      'MaxPool_3a_3x3',
      'Mixed_3b',
      'Mixed_3c',
      'MaxPool_4a_3x3',
      'Mixed_4b',
      'Mixed_4c',
      'Mixed_4d',
      'Mixed_4e',
      'Mixed_4f',
      'MaxPool_5a_2x2',
      'Mixed_5b',
      'Mixed_5c',
      'Embeddings',
  )

  def __init__(self,
               normalize_fn: Optional[types.NormalizeFn] = None,
               first_temporal_kernel_size: int = 7,
               temporal_conv_startat: Optional[str] = 'Conv2d_2c_3x3',
               gating_startat: Optional[str] = 'Conv2d_2c_3x3',
               name='S3D'):
    super().__init__(name=name)
    self._first_temporal_kernel_size = first_temporal_kernel_size
    self._temporal_conv_startat = temporal_conv_startat
    self._gating_startat = gating_startat
    self._normalize_fn = normalize_fn

    if (temporal_conv_startat not in self.VALID_ENDPOINTS
        and temporal_conv_startat is not None):
      raise ValueError(
          f'Provided `temporal_conv_startat`: {temporal_conv_startat} not '
          f'valid. It must be one of: {self.VALID_ENDPOINTS}, or `None`.')

    if (gating_startat not in self.VALID_ENDPOINTS
        and gating_startat is not None):
      raise ValueError(
          f'Provided `gating_startat`: {gating_startat} not valid. '
          f'It must be one of: {self.VALID_ENDPOINTS}, or `None`.')

    if first_temporal_kernel_size not in [1, 3, 5, 7]:
      raise ValueError('`first_temporal_kernel_size` can only be 1, 3, 5 or 7.')

  def __call__(self,
               inputs: types.TensorLike,
               is_training: bool,
               final_endpoint: str = 'Embeddings') -> jnp.ndarray:
  
    if final_endpoint not in self.VALID_ENDPOINTS:
      raise ValueError(f'Provided final_endpoint: {final_endpoint} not valid.'
                       f' It must be one of: {self.VALID_ENDPOINTS}')

    x = inputs
    layers = []
    if x.shape[-1] == 3:
      layers.append(
          _Layer('Conv2d_1a_7x7', SUnit3D,
                 dict(output_channels=64, stride=(2, 2, 2), separable=False,
                      kernel_shape=(self._first_temporal_kernel_size, 7, 7),
                      normalize_fn=self._normalize_fn)))
    else:
      layers.append(
          _Layer('Conv2d_1a_7x7', SUnit3D,
                 dict(output_channels=64, kernel_shape=(2, 4, 4),
                      stride=(1, 1, 1), separable=False,
                      normalize_fn=self._normalize_fn)))

    layers.extend([
        _Layer('MaxPool_2a_3x3', _MaxPool,
               dict(window_shape=(1, 1, 3, 3, 1), strides=(1, 1, 2, 2, 1),
                    padding='SAME')),
        _Layer('Conv2d_2b_1x1', SUnit3D,
               dict(output_channels=64, kernel_shape=(1, 1, 1),
                    normalize_fn=self._normalize_fn)),
        _Layer('Conv2d_2c_3x3', SUnit3D,
               dict(output_channels=192, separable=True,
                    normalize_fn=self._normalize_fn)),
        _Layer('MaxPool_3a_3x3', _MaxPool,
               dict(window_shape=(1, 1, 3, 3, 1), strides=(1, 1, 2, 2, 1),
                    padding='SAME')),
        _Layer('Mixed_3b', InceptionBlockV13D,
               dict(output_channels=(64, 96, 128, 16, 32, 32),
                    normalize_fn=self._normalize_fn)),
        _Layer('Mixed_3c', InceptionBlockV13D,
               dict(output_channels=(128, 128, 192, 32, 96, 64),
                    normalize_fn=self._normalize_fn)),
        _Layer('MaxPool_4a_3x3', _MaxPool,
               dict(window_shape=(1, 3, 3, 3, 1), strides=(1, 2, 2, 2, 1),
                    padding='SAME')),
        _Layer('Mixed_4b', InceptionBlockV13D,
               dict(output_channels=(192, 96, 208, 16, 48, 64),
                    normalize_fn=self._normalize_fn)),
        _Layer('Mixed_4c', InceptionBlockV13D,
               dict(output_channels=(160, 112, 224, 24, 64, 64),
                    normalize_fn=self._normalize_fn)),
        _Layer('Mixed_4d', InceptionBlockV13D,
               dict(output_channels=(128, 128, 256, 24, 64, 64),
                    normalize_fn=self._normalize_fn)),
        _Layer('Mixed_4e', InceptionBlockV13D,
               dict(output_channels=(112, 144, 288, 32, 64, 64),
                    normalize_fn=self._normalize_fn)),
        _Layer('Mixed_4f', InceptionBlockV13D,
               dict(output_channels=(256, 160, 320, 32, 128, 128),
                    normalize_fn=self._normalize_fn)),
        _Layer('MaxPool_5a_2x2', _MaxPool,
               dict(window_shape=(1, 2, 2, 2, 1), strides=(1, 2, 2, 2, 1),
                    padding='SAME')),
        _Layer('Mixed_5b', InceptionBlockV13D,
               dict(output_channels=(256, 160, 320, 32, 128, 128),
                    normalize_fn=self._normalize_fn)),
        _Layer('Mixed_5c', InceptionBlockV13D,
               dict(output_channels=(384, 192, 384, 48, 128, 128),
                    normalize_fn=self._normalize_fn)),
    ])

    self_gating_fn = None
    temporal_kernel_size = 1
    for layer in layers:
      if layer.name == self._gating_startat:
        self_gating_fn = self_gating
      if layer.name == self._temporal_conv_startat:
        temporal_kernel_size = 3

      kwargs = layer.kwargs

      if layer.module is SUnit3D:
        kwargs['self_gating_fn'] = self_gating_fn
        if 'kernel_shape' not in kwargs:
          kwargs['kernel_shape'] = (temporal_kernel_size, 3, 3)

      elif layer.module is InceptionBlockV13D:
        kwargs['self_gating_fn'] = self_gating_fn
        kwargs['temporal_kernel_size'] = temporal_kernel_size

      module = layer.module(name=layer.name, **kwargs)
      x = module(x, is_training=is_training)
      if final_endpoint == layer.name:
        return x

    assert final_endpoint == 'Embeddings'
    return jnp.mean(x, axis=(1, 2, 3))
