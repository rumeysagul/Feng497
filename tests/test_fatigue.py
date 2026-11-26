"""Yorulma eğrisi, ters interpolasyon ve çevrim sınırları."""

import unittest

import numpy as np

from feng497.fatigue import LOG_CYCLES, first_crack_peaks, life_at_peaks


class FatigueCurveTests(unittest.TestCase):
    def test_first_crack_takes_the_weakest_hole_point_with_compression_and_shear(self):
        s11 = np.array([[4.0, 0.0], [-20.0, 0.0]])
        s22 = np.array([[0.1, -10.0], [-0.1, 0.1]])
        t12 = np.array([[0.0, 4.0], [0.0, 0.0]])
        limits = first_crack_peaks(s11, s22, t12, [(100, 80, 10, 20, 5)])
        np.testing.assert_allclose(limits[:, 0], [1 / np.hypot(-0.5, 0.8), 4.0])

    def test_life_uses_log_interpolation_and_distinguishes_out_of_range_loads(self):
        curve = 2.0 - 0.1 * LOG_CYCLES
        peaks = [10**2.1, 10**2.0, 10**(2.0 - 0.1 * 4.625), 10**1.05]
        life, status = life_at_peaks(curve, peaks)
        np.testing.assert_allclose(life, [0, 0, 4.625, 9])
        self.assertEqual(status.tolist(), ["ilk_cevrim", "aralik_ici", "aralik_ici", "aralik_ustu"])
        for invalid in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                life_at_peaks(curve, invalid)

    def test_flat_peak_uses_earliest_equal_horizon_and_interpolates_after_plateau(self):
        curve = 2.0 - 0.4 * LOG_CYCLES
        curve[1:3] = 2.0
        life, _ = life_at_peaks(curve, [100, 10**1.85])
        np.testing.assert_allclose(life, [0.0, 0.625])
