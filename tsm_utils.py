from typing import Tuple

import jax
import jax.numpy as jnp

from models import types


def prepare_inputs(
    inputs: types.TensorLike) -> Tuple[jnp.ndarray, str, int]:
  if len(inputs.shape) == 5:
    tsm_mode = 'gpu'
    num_frames = inputs.shape[1]
    inputs = jnp.reshape(inputs, [-1] + list(inputs.shape[2:]))
  else:
    tsm_mode = 'tpu'
    num_frames = None
  return inputs, tsm_mode, num_frames


def prepare_outputs(outputs: types.TensorLike,
                    tsm_mode: str,
                    num_frames: int) -> jnp.ndarray:
  n_channels = outputs.shape[-1]
  if tsm_mode == 'tpu':
    outputs = jnp.reshape(outputs, [num_frames, -1, n_channels])
    outputs = jnp.mean(outputs, axis=0)
  elif tsm_mode == 'gpu':
    outputs = jnp.reshape(outputs, [-1, num_frames, n_channels])
    outputs = jnp.mean(outputs, axis=1)
  else:
    raise ValueError(
        f'`tsm_mode` should be \'tpu\' or \'gpu\' ({tsm_mode} given)')
  return outputs


def apply_temporal_shift(
    x: types.TensorLike,
    tsm_mode: str,
    num_frames: int,
    channel_shift_fraction: float = 0.125) -> jnp.ndarray:
  if tsm_mode == 'tpu':
    outputs = temporal_shift_tpu(x, num_frames, channel_shift_fraction)
  elif tsm_mode == 'gpu':
    outputs = temporal_shift_gpu(x, num_frames, channel_shift_fraction)
  else:
    raise ValueError(
        f'`tsm_mode` should be \'tpu\' or \'gpu\' ({tsm_mode} given)')
  return outputs


def temporal_shift_gpu(
    x: types.TensorLike,
    num_frames: int,
    channel_shift_fraction: float = 0.125) -> jnp.ndarray:
  orig_shp = tuple(x.shape)
  reshaped_x = jnp.reshape(x, (-1, num_frames) + orig_shp[1:])
  n_channels = orig_shp[-1]
  n_shift = int(n_channels * channel_shift_fraction)

  new_shp = tuple(reshaped_x.shape)
  shifted_backward = jax.lax.slice(
      reshaped_x, (0, 1, 0, 0, new_shp[4] - n_shift),
      (new_shp[0], new_shp[1], new_shp[2], new_shp[3], new_shp[4]))
  shifted_backward_padding = ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0))
  shifted_backward = jnp.pad(shifted_backward, shifted_backward_padding)

  shifted_forward = jax.lax.slice(
      reshaped_x, (0, 0, 0, 0, 0),
      (new_shp[0], new_shp[1] - 1, new_shp[2], new_shp[3], n_shift))
  shifted_forward_padding = ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0))
  shifted_forward = jnp.pad(shifted_forward, shifted_forward_padding)

  no_shift = reshaped_x[:, :, :, :, n_shift:-n_shift]
  shifted_x = jnp.concatenate([shifted_backward, no_shift, shifted_forward],
                              axis=4)
  return jnp.reshape(shifted_x, (-1,) + orig_shp[1:])


def temporal_shift_tpu(
    x: types.TensorLike,
    num_frames: int,
    channel_shift_fraction: float = 0.125):
  original_shape = list(x.shape)

  batch_size = int(original_shape[0] / num_frames)
  n_channels = int(original_shape[-1])
  n_shift = int(n_channels * channel_shift_fraction)

  x = x.astype(jnp.bfloat16)

  orig_shp = list(x.shape)

  shifted_backward_padding = ((0, batch_size, 0), (0, 0, 0), (0, 0, 0),
                              (0, n_channels - n_shift, 0))
  x_backward_padding = jax.lax.pad(
      x,
      padding_value=jnp.bfloat16(0.),
      padding_config=shifted_backward_padding)
  shifted_backward = jax.lax.slice(x_backward_padding,
                                   (batch_size, 0, 0, n_channels - n_shift),
                                   (orig_shp[0] + batch_size, orig_shp[1],
                                    orig_shp[2], 2 * n_channels - n_shift))
  shifted_forward_padding = ((batch_size, 0, 0), (0, 0, 0), (0, 0, 0),
                             (n_channels - n_shift, 0, 0))
  x_forward_padding = jax.lax.pad(
      x,
      padding_value=jnp.bfloat16(0.),
      padding_config=shifted_forward_padding)
  shifted_forward = jax.lax.slice(
      x_forward_padding, (0, 0, 0, 0),
      (orig_shp[0], orig_shp[1], orig_shp[2], n_channels))
  mask_noshift = (jnp.reshape((jnp.arange(n_channels) >= n_shift) &
                              (jnp.arange(n_channels) < n_channels - n_shift),
                              (1, 1, 1, -1))).astype(jnp.bfloat16)
  no_shift = mask_noshift * x
  shifted_x = shifted_backward + shifted_forward + no_shift

  return shifted_x.astype(jnp.float32)
