"""Gerilme profili dönüşümü ve yorulma hesabı arasındaki bağlantı."""

import unittest

import numpy as np
import pandas as pd

from feng497.fatigue import LOG_CYCLES, life_at_peaks
from feng497.hole_stress import THETA_DEG, fatigue_curve, fibre_stresses, shift_profiles


class UnitLoadModel:
    def __init__(self, value):
        self.value = value

    def predict(self, features):
        profile = np.zeros((len(features), len(THETA_DEG)))
        profile[:, len(THETA_DEG) // 2] = self.value
        return profile


class HoleStressTests(unittest.TestCase):
    def test_periodic_alignment_matches_angular_shift_and_round_trips(self):
        offsets = np.array([35.51, 90.0])
        original = np.broadcast_to(np.sin(np.deg2rad(3 * THETA_DEG)), (2, 360))
        shifted = shift_profiles(original, offsets)
        np.testing.assert_allclose(shifted, np.sin(np.deg2rad(3 * (THETA_DEG + offsets[:, None]))), atol=1e-13)
        np.testing.assert_allclose(shift_profiles(shifted, -offsets), original, atol=1e-13)

    def test_tangent_stress_rotates_into_fibre_components(self):
        s11, s22, t12 = fibre_stresses(np.full(360, 10.0), 0.0)
        np.testing.assert_allclose([s11[180], s22[180], t12[180]], [0, 10, 0], atol=1e-12)
        np.testing.assert_allclose([s11[225], s22[225], t12[225]], [5, 5, -5], atol=1e-12)
        np.testing.assert_allclose([s11[270], s22[270], t12[270]], [10, 0, 0], atol=1e-12)
        np.testing.assert_allclose(s11 + s22, 10.0, atol=1e-12)

    def test_stress_predictions_are_combined_before_life_calculation(self):
        transverse = 10 ** (2 - 0.1 * LOG_CYCLES)
        strengths = np.tile([1500.0, 1500.0, 0, 200.0, 70.0], (len(LOG_CYCLES), 1))
        strengths[:, 2] = transverse
        bundle = {
            "models": {"X": UnitLoadModel(1.0), "Y": UnitLoadModel(2.0)},
            "frames": {"X": "global", "Y": "global"},
            "strengths": strengths,
        }
        cases = pd.DataFrame([{"D": 30, "ex": 0, "ey": 0, "fib": 0}])
        curve = fatigue_curve(bundle, cases, "biaxial")[0]
        np.testing.assert_allclose(curve, np.log10(transverse / 3))
        log_life, status = life_at_peaks(curve, 10**1.5 / 3)
        self.assertEqual(str(status), "aralik_ici")
        self.assertAlmostEqual(float(log_life), 5)
