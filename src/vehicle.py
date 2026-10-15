"""CommonRoad rear-axle kinematic single-track experiments (SI units)."""

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks


SPEED = 5.0
TURN_STEERING = 0.2
DURATION = 10.0


def simulate(steering_angle=0.0, speed=SPEED, duration=DURATION):
    """Return sampled states [x, y, steering, speed, yaw] from rest pose.

    Speed and steering are initial conditions; inputs are zero steering rate
    and zero acceleration. CommonRoad applies its own input constraints.
    """
    p = parameters_vehicle2()
    if not np.all(np.isfinite([steering_angle, speed, duration])):
        raise ValueError("Simulation arguments must be finite")
    if duration <= 0:
        raise ValueError("Duration must be positive")
    if not p.steering.min <= steering_angle <= p.steering.max:
        raise ValueError("Steering angle outside vehicle limits")
    if not p.longitudinal.v_min <= speed <= p.longitudinal.v_max:
        raise ValueError("Speed outside vehicle limits")
    solution = solve_ivp(
        lambda t, state: vehicle_dynamics_ks(state, [0.0, 0.0], p),
        (0.0, duration),
        [0.0, 0.0, steering_angle, speed, 0.0],
        t_eval=np.linspace(0.0, duration, 1001),
        rtol=1e-10,
        atol=1e-12,
        max_step=0.05,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    if not np.isfinite(solution.y).all():
        raise RuntimeError("Nonfinite simulation output")
    return solution


def expected_radius(steering_angle):
    """Signed rear-axle radius; positive steering turns left."""
    p = parameters_vehicle2()
    if steering_angle == 0:
        return np.inf
    return (p.a + p.b) / np.tan(steering_angle)


def observed_radius(solution):
    """Fit a circle to simulated rear-axle positions by least squares."""
    x, y = solution.y[:2]
    matrix = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
    cx, cy, offset = np.linalg.lstsq(matrix, x*x + y*y, rcond=None)[0]
    return np.sqrt(offset + cx*cx + cy*cy)


def main():
    # Import only for plotting; Agg supports WSL/headless execution.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(exist_ok=True)
    for name, steering in [("straight_line", 0.0), ("constant_radius", TURN_STEERING)]:
        solution = simulate(steering)
        fig, ax = plt.subplots()
        ax.plot(solution.y[0], solution.y[1])
        ax.set(xlabel="Rear-axle x [m]", ylabel="Rear-axle y [m]",
               title=f"{name.replace('_', ' ').title()}: v={SPEED:g} m/s, steering={steering:g} rad")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}.png", dpi=150)
        plt.close(fig)
        if steering:
            print(f"Expected radius: {expected_radius(steering):.10f} m")
            print(f"Observed radius: {observed_radius(solution):.10f} m")
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
