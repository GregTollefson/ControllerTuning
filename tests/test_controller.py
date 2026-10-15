"""Sign, constraint, termination, and broad closed-loop performance checks."""
import unittest

import numpy as np

from src.controller import StanleyController
from src.path import ReferencePath
from src.tracking import simulate_tracking
from src.vehicle import parameters_vehicle2


class TestController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = simulate_tracking()
        cls.parameters = parameters_vehicle2()

    def test_aligned_straight_path(self):
        command = StanleyController().command(
            [0, 0, 0, 5, 0], ReferencePath(amplitude=0))
        np.testing.assert_allclose(command, 0, atol=1e-12)

    def test_offset_steering_direction(self):
        for offset in [-0.5, 0.5]:
            desired, rate, cte, heading = StanleyController().command(
                [0, offset, 0, 5, 0], ReferencePath(amplitude=0))
            self.assertLess(desired * offset, 0)
            self.assertLess(rate * offset, 0)
            self.assertAlmostEqual(cte, -offset)
            self.assertAlmostEqual(heading, 0)

    def test_front_axle_geometry(self):
        p = self.parameters
        yaw = 0.1
        state = [0, 0, 0, 5, yaw]
        _, _, error, heading = StanleyController().command(state, ReferencePath(amplitude=0))
        self.assertAlmostEqual(error, -(p.a + p.b) * np.sin(yaw))
        self.assertAlmostEqual(heading, -yaw)

    def test_vehicle_limits(self):
        controller = StanleyController()
        path = ReferencePath(amplitude=0)
        limits = self.parameters.steering
        # Large errors force desired angle saturation and rate clipping.
        for offset, expected_angle in [(-100, limits.max), (100, limits.min)]:
            desired, rate, _, _ = controller.command([0, offset, 0, 5, 0], path)
            self.assertEqual(desired, expected_angle)
            self.assertGreaterEqual(rate, limits.v_min)
            self.assertLessEqual(rate, limits.v_max)
            outward = controller.command([0, offset, expected_angle, 5, 0], path)[1]
            self.assertEqual(outward, 0)
        r = self.result
        for angles in [r.desired_steering, r.states[2]]:
            self.assertTrue(np.all(angles >= limits.min - 1e-8))
            self.assertTrue(np.all(angles <= limits.max + 1e-8))
        self.assertTrue(np.all(r.steering_rate >= limits.v_min))
        self.assertTrue(np.all(r.steering_rate <= limits.v_max))

    def test_finite_simulation_and_constant_speed(self):
        r = self.result
        for values in [r.time, r.states, r.cte, r.heading_error,
                       r.desired_steering, r.steering_rate]:
            self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.all(np.diff(r.time) > 0))
        np.testing.assert_allclose(r.states[3], 5, rtol=0, atol=1e-12)
        self.assertEqual(r.termination, "duration")
        self.assertAlmostEqual(r.time[-1], 20)

    def test_curved_tracking_performance(self):
        r = self.result
        # Broad physical tolerances: RMS < 0.224 m, no large departures,
        # and small residual errors after the initial correction.
        self.assertLess(r.mse, 0.05)
        self.assertLess(r.max_error, 0.75)
        self.assertLess(np.max(np.abs(r.cte[r.time > 5])), 0.15)
        self.assertGreater(abs(r.cte[0]), 0.4)

    def test_path_end_termination(self):
        path = ReferencePath(length=15)
        r = simulate_tracking(path=path)
        self.assertEqual(r.termination, "path_end")
        self.assertLess(r.time[-1], 20)
        p = self.parameters
        state = r.states[:, -1]
        front = state[:2] + (p.a+p.b)*np.array([np.cos(state[4]), np.sin(state[4])])
        heading = path.heading(path.length)
        endpoint_distance = np.dot(front - [path.length, path.y(path.length)],
                                   [np.cos(heading), np.sin(heading)])
        self.assertAlmostEqual(endpoint_distance, 0, delta=1e-7)


if __name__ == "__main__":
    unittest.main()
