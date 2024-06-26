from absl.testing import absltest
from absl.testing import parameterized

import haiku as hk
import jax
import jax.numpy as jnp

from models import tsm_resnet


class TSMResNetTest(parameterized.TestCase):

  @parameterized.parameters(
      ('tsm_resnet_stem', (2 * 32, 56, 56, 64)),
      ('tsm_resnet_unit_0', (2 * 32, 56, 56, 256)),
      ('tsm_resnet_unit_1', (2 * 32, 28, 28, 512)),
      ('tsm_resnet_unit_2', (2 * 32, 14, 14, 1024)),
      ('tsm_resnet_unit_3', (2 * 32, 7, 7, 2048)),
      ('last_conv', (2 * 32, 7, 7, 2048)),
      ('Embeddings', (2, 2048)),
  )
  def test_output_dimension(self, final_endpoint, expected_shape):
    input_shape = (2, 32, 224, 224, 3)

    def f():
      data = jnp.zeros(input_shape)
      net = tsm_resnet.TSMResNetV2()
      return net(data, final_endpoint=final_endpoint)

    init_fn, apply_fn = hk.transform(f)
    out = apply_fn(init_fn(jax.random.PRNGKey(42)), None)
    self.assertEqual(out.shape, expected_shape)

  def test_tpu_mode(self):
    input_shape = (32 * 2, 224, 224, 3)

    def f():
      data = jnp.zeros(input_shape)
      net = tsm_resnet.TSMResNetV2(num_frames=32)
      return net(data, final_endpoint='Embeddings')

    init_fn, apply_fn = hk.transform(f)
    out = apply_fn(init_fn(jax.random.PRNGKey(42)), None)
    self.assertEqual(out.shape, (2, 2048))


if __name__ == '__main__':
  absltest.main()
