"""Deterministic continuously evaluated closed-loop CommonRoad KS simulation."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from src.controller import StanleyController
from src.path import ReferencePath
from src.vehicle import SPEED, parameters_vehicle2, vehicle_dynamics_ks


@dataclass
class TrackingResult:
    time: np.ndarray
    states: np.ndarray  # rows: x, y, steering, speed, yaw
    desired_steering: np.ndarray
    steering_rate: np.ndarray
    cte: np.ndarray
    heading_error: np.ndarray
    termination: str

    @property
    def mse(self):
        return float(np.mean(self.cte**2))

    @property
    def max_error(self):
        return float(np.max(np.abs(self.cte)))

    @property
    def limits_reached(self):
        p = parameters_vehicle2().steering
        return {
            "angle": bool(np.any(self.states[2] <= p.min + 1e-7)
                          or np.any(self.states[2] >= p.max - 1e-7)
                          or np.any(self.desired_steering <= p.min + 1e-7)
                          or np.any(self.desired_steering >= p.max - 1e-7)),
            "rate": bool(np.any(self.steering_rate <= p.v_min + 1e-7)
                         or np.any(self.steering_rate >= p.v_max - 1e-7)),
        }


def simulate_tracking(path=None, controller=None, duration=20.0, speed=SPEED, offset=0.5):
    path = path if path is not None else ReferencePath()
    controller = controller if controller is not None else StanleyController()
    p = parameters_vehicle2()
    if not np.all(np.isfinite([duration, speed, offset])) or duration <= 0:
        raise ValueError("Finite arguments and positive duration required")
    if not 0 < speed <= p.longitudinal.v_max:
        raise ValueError("Tracking requires a positive speed within vehicle limits")
    # Front axle starts near x=L; rear axle is offset in world y at x=0.
    initial = [0.0, path.y(0.0) + offset, 0.0, speed, path.heading(0.0)]

    def dynamics(t, state):
        rate = controller.command(state, path, p)[1]
        return vehicle_dynamics_ks(state, [rate, 0.0], p)

    def path_end(t, state):
        # End when the front axle crosses the endpoint's normal plane.
        yaw = state[4]
        front = state[:2] + (p.a + p.b) * np.array([np.cos(yaw), np.sin(yaw)])
        tangent = np.array([np.cos(path.heading(path.length)), np.sin(path.heading(path.length))])
        return float(np.dot(front - [path.length, path.y(path.length)], tangent))

    path_end.terminal = True
    path_end.direction = 1
    if path_end(0, np.asarray(initial)) >= 0:
        raise ValueError("Path must extend beyond initial front axle")
    solution = solve_ivp(dynamics, (0, duration), initial, events=path_end,
                         dense_output=True, rtol=1e-9, atol=1e-11, max_step=0.05)
    if not solution.success:
        raise RuntimeError(solution.message)
    # Uniform samples, including the exact termination time (roughly 50 Hz).
    times = np.linspace(0, solution.t[-1], int(np.ceil(solution.t[-1] / 0.02)) + 1)
    states = solution.sol(times)
    commands = np.array([controller.command(state, path, p) for state in states.T]).T
    if not np.isfinite(states).all() or not np.isfinite(commands).all():
        raise RuntimeError("Nonfinite tracking output")
    return TrackingResult(times, states, *commands,
                          "path_end" if solution.t_events[0].size else "duration")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = ReferencePath()
    result = simulate_tracking(path)
    output = Path(__file__).resolve().parents[1] / "results"
    output.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.linspace(0, path.length, 1201)
    ax.plot(x, path.y(x), "--", label="Reference (front-axle target)")
    ax.plot(*result.states[:2], label="Rear axle")
    p = parameters_vehicle2()
    ax.plot(result.states[0] + (p.a+p.b)*np.cos(result.states[4]),
            result.states[1] + (p.a+p.b)*np.sin(result.states[4]), label="Front axle")
    ax.set(xlabel="x [m]", ylabel="y [m]", title="Stanley path tracking")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output / "tracking_trajectory.png", dpi=150)
    plt.close(fig)
    for name, ylabel, series in [
        ("tracking_error", "Front-axle CTE [m]", [(result.cte, "CTE")]),
        ("tracking_steering", "Steering angle [rad]", [
            (result.states[2], "Actual angle"), (result.desired_steering, "Desired angle")])]:
        fig, ax = plt.subplots()
        for values, label in series:
            ax.plot(result.time, values, label=label)
        ax.set(xlabel="Time [s]", ylabel=ylabel)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=150)
        plt.close(fig)
    print(f"Stanley gain=1.0 /s; actuator gain=5.0 /s; speed={SPEED:g} m/s")
    print(f"CTE MSE={result.mse:.8f} m^2; max |CTE|={result.max_error:.8f} m")
    print(f"Limits reached: {result.limits_reached}; termination={result.termination}")


if __name__ == "__main__":
    main()
