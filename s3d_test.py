from absl.testing import absltest
from absl.testing import parameterized

import haiku as hk
import jax
import numpy as np

from models import normalization
from models import s3d


class _CallableS3D:

  def __init__(self, *args, **kwargs):
    self._model = hk.transform_with_state(
        lambda *a, **k:  
        s3d.S3D(
            normalize_fn=normalization.get_normalize_fn(),
            *args, **kwargs)(*a, **k))
    self._rng = jax.random.PRNGKey(42)
    self._params, self._state = None, None

  def init(self, inputs, **kwargs):
    self._params, self._state = self._model.init(
        self._rng, inputs, is_training=True, **kwargs)

  def __call__(self, inputs, **kwargs):
    if self._params is None:
      self.init(inputs)
    output, _ = self._model.apply(
        self._params, self._state, self._rng, inputs, **kwargs)
    return output


class S3DTest(parameterized.TestCase):
  @parameterized.parameters(
      dict(endpoint='Embeddings', expected_size=(2, 1024)),
  )
  def test_endpoint_expected_output_dimensions(self, endpoint, expected_size):
    inputs = np.random.normal(size=(2, 16, 224, 224, 3))
    model = _CallableS3D()
    output = model(inputs, is_training=False, final_endpoint=endpoint)
    self.assertSameElements(output.shape, expected_size)

  def test_space_to_depth(self):
    inputs = np.random.normal(size=(2, 16//2, 224//2, 224//2, 3*2*2*2))
    model = _CallableS3D()
    output = model(inputs, is_training=False, final_endpoint='Conv2d_1a_7x7')
    self.assertSameElements(output.shape, (2, 8, 112, 112, 64))

if __name__ == '__main__':
  absltest.main()
