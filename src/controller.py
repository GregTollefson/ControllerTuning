"""Fixed-gain Stanley steering with a proportional steering-rate actuator."""
from dataclasses import dataclass

import numpy as np
from vehiclemodels.utils.steering_constraints import steering_constraints

from src.vehicle import parameters_vehicle2


@dataclass(frozen=True)
class StanleyController:
    gain: float = 1.0  # 1/s
    actuator_gain: float = 5.0  # 1/s, fixed

    def command(self, state, path, parameters=None):
        """Return (desired angle, constrained rate, front-axle CTE, heading error)."""
        p = parameters if parameters is not None else parameters_vehicle2()
        x, y, steering, speed, yaw = state
        front_x = x + (p.a + p.b) * np.cos(yaw)
        front_y = y + (p.a + p.b) * np.sin(yaw)
        heading, error = path.tracking_error(front_x, front_y)
        heading_error = np.arctan2(np.sin(heading - yaw), np.cos(heading - yaw))
        # Positive error means path left of front axle: command positive steering.
        desired = np.clip(heading_error + np.arctan2(self.gain * error, max(speed, 1e-3)),
                          p.steering.min, p.steering.max)
        rate = steering_constraints(steering, self.actuator_gain * (desired - steering), p.steering)
        return float(desired), float(rate), error, float(heading_error)
