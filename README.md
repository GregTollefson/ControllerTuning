# Milestone 1: CommonRoad vehicle simulation

Run from the project root with the existing virtual environment:

```bash
MPLCONFIGDIR=/tmp/controller-tuning-mpl .venv/bin/python -m src.vehicle
.venv/bin/python -m unittest discover -s tests -v
```

The simulation uses CommonRoad's actual `vehicle_dynamics_ks` and
`parameters_vehicle2()` (vehicle ID 2, BMW 320i). All units are SI.
States are `[x, y, steering angle, speed, yaw]`, with position at the rear axle.
Controls are `[steering rate, longitudinal acceleration]`, both held at zero.
The steering angle is initialized directly; there is no steering transient.

Both deterministic experiments start at the origin with zero yaw and run for
10 seconds at 5 m/s. Straight driving uses zero steering and ends at (50, 0) m.
Turning uses 0.2 rad steering. Its rear-axle radius is
`(a + b) / tan(steering)`, approximately 12.7222 m, with center at (0, R).
The program prints the analytic radius and a circle fit to the trajectory.
Plots are saved as `results/straight_line.png` and `results/constant_radius.png`.

SciPy `solve_ivp` uses `rtol=1e-10`, `atol=1e-12`, a maximum step of 0.05 s,
and 1001 output samples. Tests check analytic straight and circular paths,
constant speed and steering, and finite outputs. No extra dependencies are needed.

This is an ideal kinematic model without tire slip, road interaction, or actuator
transients. CommonRoad's built-in input constraints remain active. No controller,
Monte Carlo simulation, or optimization is included.
