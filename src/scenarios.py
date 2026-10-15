"""Reusable independent driving scenarios; offsets are left-normal rear-axle offsets."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    speed: float  # m/s
    offset: float  # m, positive left of path at its start


def generate_scenarios(count=20, seed=123):
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("Scenario count must be a positive integer")
    rng = np.random.default_rng(seed)
    speeds = rng.uniform(4.0, 8.0, count)
    offsets = rng.uniform(-0.5, 0.5, count)
    return tuple(Scenario(i, float(speed), float(offset))
                 for i, (speed, offset) in enumerate(zip(speeds, offsets), start=1))
