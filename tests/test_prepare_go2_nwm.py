import unittest

import numpy as np

from scripts.prepare_go2_nwm import local_deltas, nearest_indices


class Go2ConversionTest(unittest.TestCase):
    def test_receive_timestamps_select_real_paired_rows(self):
        # Host time and video frame/15 time differ; choose with host time.
        times = 5.0 + np.arange(20) * .070
        indices, grid = nearest_indices(times, 4)
        np.testing.assert_allclose(grid, [5, 5.25, 5.5, 5.75, 6, 6.25])
        np.testing.assert_array_equal(indices, [0, 4, 7, 11, 14, 18])

    def test_bad_timestamp_sequences_are_rejected(self):
        for times in ([0, 0, .1], [0, -.1], [0, np.nan], []):
            with self.assertRaises(ValueError):
                nearest_indices(times, 4)

    def test_no_duplicate_frame_upsampling(self):
        with self.assertRaises(ValueError):
            nearest_indices([0, 1, 2], 4)

    def test_local_forward_left_and_wrapped_yaw(self):
        # Facing +world-Y, +world-Y is forward and -world-X is left.
        xy = np.array([[0, 0], [0, 1], [-1, 1]], dtype=float)
        result = local_deltas(xy, np.full(3, np.pi / 2))
        np.testing.assert_allclose(result, [[1, 0, 0], [0, 1, 0]], atol=1e-12)
        result = local_deltas(np.zeros((2, 2)), np.array([np.pi - .1, -np.pi + .1]))
        self.assertAlmostEqual(result[0, 2], .2)


if __name__ == '__main__':
    unittest.main()
