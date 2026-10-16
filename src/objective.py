"""Cached scalar objective with an explicit loss for every input scenario."""
import numpy as np

from src.monte_carlo import evaluate_gain


FAILURE_PENALTY = 1.0  # m²: 1 m RMS, far above nominal tracking losses


def penalized_losses(evaluation, penalty=FAILURE_PENALTY):
    """Failed, incomplete, or invalid-loss runs receive a finite 1 m² penalty."""
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("Penalty must be finite and positive")
    return np.array([
        float(r.mse) if r.status == "completed" and r.mse is not None
        and np.isfinite(r.mse) and r.mse >= 0 else float(penalty)
        for r in evaluation.results
    ])


class TrackingObjective:
    """Fix scenarios and simulation settings once; cache exact floating-point gains."""
    def __init__(self, scenarios, penalty=FAILURE_PENALTY, endpoint=100.0, max_time=40.0):
        self.scenarios = tuple(scenarios)
        if not self.scenarios or not np.isfinite(penalty) or penalty <= 0:
            raise ValueError("Nonempty scenarios and positive finite penalty required")
        self.penalty = penalty
        self.endpoint = endpoint
        self.max_time = max_time
        self.cache = {}
        self.vehicle_simulations = 0

    def __call__(self, k):
        k = float(k)
        if not np.isfinite(k) or k < 0:
            raise ValueError("Gain must be finite and nonnegative")
        if k not in self.cache:
            evaluation = evaluate_gain(k, self.scenarios, self.endpoint, self.max_time)
            if len(evaluation.results) != len(self.scenarios):
                raise RuntimeError("Evaluation omitted scenarios")
            losses = penalized_losses(evaluation, self.penalty)
            # Divide first to avoid overflow even for very large finite losses.
            value = float(np.sum(losses / len(losses)))
            if not np.isfinite(value):
                value = float(np.max(losses))
            self.cache[k] = (value, evaluation)
            self.vehicle_simulations += len(self.scenarios)
        return self.cache[k][0]

    def diagnostics(self, k):
        self(k)
        return self.cache[float(k)][1]
