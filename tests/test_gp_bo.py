"""Numerical and budget checks using cheap synthetic objectives and mocked runs."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import norm

from src.gp_bo import (
    BOUNDS, DUPLICATE_TOLERANCE, GaussianProcess, expected_improvement,
    optimize_bo, random_search,
)
from src.objective import TrackingObjective, penalized_losses
from src.scenarios import generate_scenarios


def synthetic(k):
    return (k - 1.37)**2 + 0.02


class TestGPBO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls = []

        def counted(k):
            cls.calls.append(k)
            return synthetic(k)

        cls.bo = optimize_bo(counted, budget=10, seed=12, grid_size=601)
        cls.random = random_search(synthetic, budget=10, seed=789)

    def test_analytic_expected_improvement(self):
        result = expected_improvement([1, 0], [1, 1], best=1, xi=0)
        np.testing.assert_allclose(result, [1/np.sqrt(2*np.pi), norm.cdf(1) + norm.pdf(1)],
                                   rtol=1e-12)

    def test_ei_finite_nonnegative_zero_variance(self):
        ei = expected_improvement([0, 1, -1, 1000], [0, 1e-15, 1, 0.1], best=0, xi=1e-5)
        self.assertTrue(np.isfinite(ei).all())
        self.assertTrue((ei >= 0).all())
        np.testing.assert_array_equal(ei[:2], 0)
        self.assertTrue(np.isfinite(expected_improvement([-1e308], [1], best=1e308)).all())
        with self.assertRaises(ValueError):
            expected_improvement([0], [1], 0, xi=-1)

    def test_gp_synthetic_prediction(self):
        x = np.linspace(*BOUNDS, 15)
        function = lambda k: np.sin(2*k)
        gp = GaussianProcess(seed=42).fit(x, function(x))
        query = np.linspace(*BOUNDS, 51)
        mean, std = gp.predict(query)
        self.assertLess(np.max(np.abs(mean - function(query))), 0.08)
        self.assertTrue(np.isfinite(mean).all())
        self.assertTrue(np.isfinite(std).all())
        self.assertTrue(np.all(std >= 0))
        self.assertAlmostEqual(gp.scale([BOUNDS[0]])[0, 0], 0)
        self.assertAlmostEqual(gp.scale([BOUNDS[1]])[0, 0], 1)

    def test_reproducibility(self):
        repeated = optimize_bo(synthetic, budget=10, seed=12, grid_size=601)
        self.assertEqual(self.bo.history, repeated.history)
        self.assertEqual(self.random.history, random_search(synthetic, budget=10, seed=789).history)
        self.assertNotEqual(self.random.history, random_search(synthetic, budget=10, seed=790).history)

    def test_bounds_duplicates_budget_and_cumulative_best(self):
        self.assertEqual(len(self.calls), 10)
        for result in [self.bo, self.random]:
            self.assertEqual(len(result.history), 10)
            gains = np.array([r.gain for r in result.history])
            self.assertTrue(np.all((gains >= BOUNDS[0]) & (gains <= BOUNDS[1])))
            self.assertTrue(np.all(np.diff(np.sort(gains)) > DUPLICATE_TOLERANCE))
            np.testing.assert_allclose(gains[:5], np.linspace(*BOUNDS, 5))
            values = [r.objective for r in result.history]
            np.testing.assert_allclose([r.cumulative_best for r in result.history],
                                       np.minimum.accumulate(values))
            self.assertEqual([r.evaluation for r in result.history], list(range(1, 11)))
        self.assertLess(self.bo.best.objective, 0.03)

    def test_default_budget(self):
        # 25 distinct evaluations even for an objective that is perfectly flat.
        result = optimize_bo(lambda k: 1.0, grid_size=101)
        self.assertEqual(len(result.history), 25)
        self.assertEqual(len(set(r.gain for r in result.history)), 25)

    def test_penalties_and_cached_diagnostics(self):
        results = [SimpleNamespace(status="completed", mse=0.01),
                   SimpleNamespace(status="incomplete", mse=0.001),
                   SimpleNamespace(status="failed", mse=None),
                   SimpleNamespace(status="completed", mse=np.nan)]
        evaluation = SimpleNamespace(results=results, statistics={"completed": 1, "incomplete": 1, "failed": 1})
        np.testing.assert_allclose(penalized_losses(evaluation), [0.01, 1, 1, 1])
        with patch("src.objective.evaluate_gain", return_value=evaluation) as evaluator:
            objective = TrackingObjective(generate_scenarios(4))
            self.assertAlmostEqual(objective(1), 3.01 / 4)
            self.assertEqual(objective(1), objective(1.0))
            self.assertIs(objective.diagnostics(1), evaluation)
            self.assertEqual(evaluator.call_count, 1)
            self.assertEqual(objective.vehicle_simulations, 4)
            objective(1.1)
            self.assertEqual(evaluator.call_count, 2)
        for status in ["incomplete", "failed"]:
            all_failed = SimpleNamespace(results=[SimpleNamespace(status=status, mse=None)])
            self.assertEqual(penalized_losses(all_failed)[0], 1)
            with patch("src.objective.evaluate_gain", return_value=all_failed):
                self.assertEqual(TrackingObjective(generate_scenarios(1))(1), 1)

    def test_invalid_search_settings(self):
        with self.assertRaises(ValueError):
            optimize_bo(synthetic, budget=4)
        with self.assertRaises(ValueError):
            random_search(synthetic, bounds=(3, 0.1))
        with self.assertRaises(ValueError):
            optimize_bo(lambda k: np.nan)


if __name__ == "__main__":
    unittest.main()
