"""Seeded sampling, statistical summaries, and explicit completion accounting."""
from dataclasses import replace
import unittest

import numpy as np

from src.monte_carlo import evaluate_gain, summarize
from src.path import ReferencePath
from src.scenarios import Scenario, generate_scenarios
from src.tracking import simulate_tracking


class TestMonteCarlo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = generate_scenarios(3)
        cls.evaluation = evaluate_gain(1.0, cls.scenarios, endpoint=15, max_time=8)

    def test_seed_reproducibility(self):
        self.assertEqual(generate_scenarios(), generate_scenarios(seed=123))
        self.assertNotEqual(generate_scenarios(), generate_scenarios(seed=124))

    def test_sampling_bounds_and_ids(self):
        scenarios = generate_scenarios(1000)
        self.assertEqual(len({s.scenario_id for s in scenarios}), len(scenarios))
        self.assertEqual(len(generate_scenarios()), 20)
        for s in scenarios:
            self.assertTrue(4 <= s.speed < 8)
            self.assertTrue(-0.5 <= s.offset < 0.5)

    def test_reusable_scenarios(self):
        repeated = evaluate_gain(1.0, self.scenarios, endpoint=15, max_time=8)
        self.assertEqual(self.evaluation.results, repeated.results)
        self.assertEqual(self.evaluation.statistics, repeated.statistics)
        different_gain = evaluate_gain(0.8, self.scenarios, endpoint=15, max_time=8)
        self.assertEqual([(r.scenario_id, r.speed, r.offset) for r in repeated.results],
                         [(r.scenario_id, r.speed, r.offset) for r in different_gain.results])

    def test_summary_formulas(self):
        base = self.evaluation.results[0]
        rows = tuple(replace(base, scenario_id=i, mse=loss, max_error=loss)
                     for i, loss in enumerate([1.0, 2.0, 3.0]))
        # Large incomplete/failed losses must not enter completed-run summaries.
        rows += (replace(base, status="incomplete", mse=1000),
                 replace(base, status="failed", mse=None, angle_saturated=None,
                         rate_saturated=None))
        stats = summarize(rows)
        self.assertEqual(stats["mean_mse"], 2)
        self.assertEqual(stats["sample_sd_mse"], 1)
        self.assertAlmostEqual(stats["mc_standard_error"], 1 / np.sqrt(3))
        self.assertAlmostEqual(stats["tracking_rmse"], np.sqrt(2))
        self.assertEqual(stats["max_abs_cte"], 3)
        self.assertEqual(stats["completed"], 3)
        self.assertEqual(stats["incomplete"], 1)
        self.assertEqual(stats["failed"], 1)
        self.assertIsNone(summarize([base])["sample_sd_mse"])
        self.assertIsNone(summarize([])["mean_mse"])

    def test_completed_results(self):
        e = self.evaluation
        self.assertEqual(e.statistics["completed"], len(self.scenarios))
        for r in e.results:
            self.assertEqual(r.status, "completed")
            self.assertTrue(np.isfinite(r.mse))
            self.assertGreaterEqual(r.mse, 0)
            self.assertTrue(np.isfinite(r.max_error))
            self.assertLess(r.elapsed_simulation, 8)
            self.assertEqual(r.endpoint, 15)

    def test_incomplete_and_failed_runs_retained(self):
        e = evaluate_gain(1, [Scenario(1, 5, 0), Scenario(2, -1, 0)],
                          endpoint=15, max_time=0.02)
        self.assertEqual(len(e.results), 2)
        self.assertEqual(e.results[0].status, "incomplete")
        self.assertIsNotNone(e.results[0].mse)
        self.assertEqual(e.results[1].status, "failed")
        self.assertIsNone(e.results[1].mse)
        self.assertTrue(all(r.message for r in e.results))
        self.assertEqual(e.statistics["completed"], 0)
        self.assertIsNone(e.statistics["mc_standard_error"])

    def test_normal_offset_initialization_and_endpoint(self):
        path = ReferencePath()
        run = simulate_tracking(path=path, k=1, offset=-0.4,
                                offset_frame="path_normal", endpoint=15)
        heading = path.heading(0)
        normal = np.array([-np.sin(heading), np.cos(heading)])
        np.testing.assert_allclose(run.states[:2, 0], -0.4 * normal, atol=1e-12)
        self.assertEqual(run.termination, "path_end")
        # The actual endpoint event, not duration, terminates a successful run.
        from src.vehicle import parameters_vehicle2
        p = parameters_vehicle2()
        yaw = run.states[4, -1]
        front = run.states[:2, -1] + (p.a + p.b) * np.array([np.cos(yaw), np.sin(yaw)])
        tangent = [np.cos(path.heading(15)), np.sin(path.heading(15))]
        self.assertAlmostEqual(np.dot(front - [15, path.y(15)], tangent), 0, delta=1e-7)


if __name__ == "__main__":
    unittest.main()
