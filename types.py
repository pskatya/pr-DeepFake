from typing import Callable, Tuple, Union

import jax.numpy as jnp
import numpy as np
import optax

TensorLike = Union[np.ndarray, jnp.DeviceArray]

ActivationFn = Callable[[TensorLike], TensorLike]
GatingFn = Callable[[TensorLike], TensorLike]
NetworkFn = Callable[[TensorLike], TensorLike]
NormalizeFn = Callable[..., TensorLike]

OptState = Tuple[optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState]


