from typing import Any, Dict, Optional, Sequence, Union

import haiku as hk
from jax import numpy as jnp

from models import types


class _BatchNorm(hk.BatchNorm):
  def __init__(self,
               create_scale: bool = True,
               create_offset: bool = True,
               decay_rate: float = 0.9,
               eps: float = 1e-5,
               test_local_stats: bool = False,
               **kwargs):
    self._test_local_stats = test_local_stats
    super().__init__(create_scale=create_scale,
                     create_offset=create_offset,
                     decay_rate=decay_rate,
                     eps=eps,
                     **kwargs)

  def __call__(self,
               x: types.TensorLike,
               is_training: bool) -> jnp.ndarray:
    return super().__call__(x, is_training,
                            test_local_stats=self._test_local_stats)


class _CrossReplicaBatchNorm(hk.BatchNorm):


  def __init__(self,
               create_scale: bool = True,
               create_offset: bool = True,
               decay_rate: float = 0.9,
               eps: float = 1e-5,
               test_local_stats: bool = False,
               **kwargs):
    self._test_local_stats = test_local_stats
    kwargs['cross_replica_axis'] = kwargs.get('cross_replica_axis', 'i')
    super().__init__(create_scale=create_scale,
                     create_offset=create_offset,
                     decay_rate=decay_rate,
                     eps=eps,
                     **kwargs)

  def __call__(self,
               x: types.TensorLike,
               is_training: bool) -> jnp.ndarray:
    return super().__call__(x, is_training,
                            test_local_stats=self._test_local_stats)


class _LayerNorm(hk.LayerNorm):

  def __init__(self,
               axis: Union[int, Sequence[int]] = (1, 2),
               create_scale: bool = True,
               create_offset: bool = True,
               **kwargs):
    super().__init__(axis=axis,
                     create_scale=create_scale,
                     create_offset=create_offset,
                     **kwargs)

  def __call__(self,
               x: types.TensorLike,
               is_training: bool) -> jnp.ndarray:
    del is_training  
    return super().__call__(x)


_NORMALIZER_NAME_TO_CLASS = {
    'batch_norm': _BatchNorm,
    'cross_replica_batch_norm': _CrossReplicaBatchNorm,
    'layer_norm': _LayerNorm,
}


def get_normalize_fn(
    normalizer_name: str = 'batch_norm',
    normalizer_kwargs: Optional[Dict[str, Any]] = None,
) -> types.NormalizeFn:
  normalizer_class = _NORMALIZER_NAME_TO_CLASS[normalizer_name]
  normalizer_kwargs = normalizer_kwargs or dict()

  return lambda *a, **k: normalizer_class(**normalizer_kwargs)(*a, **k)  
