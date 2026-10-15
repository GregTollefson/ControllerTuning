"""Numerical checks against analytic KS trajectories."""

import unittest

import numpy as np

from src.vehicle import (
    SPEED, TURN_STEERING, expected_radius, observed_radius, simulate,
)


class TestVehicle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.straight = simulate()
        cls.turn = simulate(TURN_STEERING)

    def test_straight_line(self):
        np.testing.assert_allclose(self.straight.y[0], SPEED * self.straight.t, atol=1e-8)
        np.testing.assert_allclose(self.straight.y[[1, 2, 4]], 0.0, atol=1e-12)

    def test_constant_speed_and_steering(self):
        for solution, steering in [(self.straight, 0.0), (self.turn, TURN_STEERING)]:
            np.testing.assert_allclose(solution.y[3], SPEED, rtol=0, atol=1e-12)
            np.testing.assert_allclose(solution.y[2], steering, rtol=0, atol=1e-12)

    def test_turning_radius(self):
        radius = expected_radius(TURN_STEERING)
        self.assertAlmostEqual(observed_radius(self.turn), radius, delta=1e-7)
        distances = np.hypot(self.turn.y[0], self.turn.y[1] - radius)
        np.testing.assert_allclose(distances, radius, rtol=0, atol=1e-7)
        yaw = SPEED * self.turn.t / radius
        np.testing.assert_allclose(self.turn.y[4], yaw, rtol=0, atol=1e-9)
        np.testing.assert_allclose(self.turn.y[0], radius * np.sin(yaw), atol=1e-7)
        np.testing.assert_allclose(self.turn.y[1], radius * (1 - np.cos(yaw)), atol=1e-7)

    def test_finite_outputs(self):
        for solution in [self.straight, self.turn]:
            self.assertTrue(solution.success)
            self.assertTrue(np.isfinite(solution.t).all())
            self.assertTrue(np.isfinite(solution.y).all())


if __name__ == "__main__":
    unittest.main()
