
from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as jnp
import numpy as np

from models import tsm_utils


class TsmUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      ((2, 32, 224, 224, 3), 'gpu', (2 * 32, 224, 224, 3), 32),
      ((32, 224, 224, 3), 'tpu', (32, 224, 224, 3), None),
  )
  def test_prepare_inputs(self, input_shape, expected_mode, expected_shape,
                          expected_num_frames):

    data = jnp.zeros(input_shape)
    out, mode, num_frames = tsm_utils.prepare_inputs(data)
    self.assertEqual(out.shape, expected_shape)
    self.assertEqual(mode, expected_mode)
    self.assertEqual(num_frames, expected_num_frames)

  def test_prepare_outputs(self):
    data = jnp.concatenate([jnp.zeros(4), jnp.ones(4)]).reshape(4, 2)
    out_gpu = tsm_utils.prepare_outputs(data, 'gpu', 2)
    out_tpu = tsm_utils.prepare_outputs(data, 'tpu', 2)
    expected_gpu = np.concatenate([np.zeros(2), np.ones(2)]).reshape(2, 2)
    expected_tpu = 0.5 * jnp.ones((2, 2))
    np.testing.assert_allclose(out_gpu, expected_gpu)
    np.testing.assert_allclose(out_tpu, expected_tpu)

  def test_apply_tsm(self):
    shape = (32, 224, 224, 16)
    data = jnp.zeros(shape)
    out_gpu = tsm_utils.apply_temporal_shift(data, 'gpu', 16)
    out_tpu = tsm_utils.apply_temporal_shift(data, 'tpu', 16)
    self.assertEqual(out_gpu.shape, shape)
    self.assertEqual(out_tpu.shape, shape)

if __name__ == '__main__':
  absltest.main()
