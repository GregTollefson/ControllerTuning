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

# Milestone 2: Stanley lateral tracking

```bash
MPLCONFIGDIR=/tmp/controller-tuning-mpl .venv/bin/python -m src.tracking
.venv/bin/python -m unittest discover -s tests -v
```

`src/path.py` defines the finite centerline `y(x) = 3 sin(2πx/60)` for
`0 <= x <= 120 m`. Travel is in increasing x. World y points left, and yaw and
steering angles are positive counterclockwise. The reference heading is
`atan(dy/dx)`. A coarse nearest-point search followed by continuous local
minimization avoids steering jumps from discrete waypoint selection.

`src/controller.py` computes the front axle from the rear-axle state:

```text
L = a + b = 2.5789128 m
p_front = [x + L cos(yaw), y + L sin(yaw)]
normal_left = [-sin(reference_heading), cos(reference_heading)]
e_CTE = dot(reference_point - p_front, normal_left)
e_heading = wrap_to_pi(reference_heading - yaw)
delta_desired = clip(e_heading + atan2(k * e_CTE, v), delta_min, delta_max)
steering_rate = constrain(K_actuator * (delta_desired - delta_actual))
```

Thus an axle left of the path has negative CTE and receives a rightward
correction. Stanley gain `k = 1.0 /s` and actuator gain `K_actuator = 5.0 /s`
are fixed. The desired steering **angle** is in radians; the CommonRoad control
input is steering **rate** in rad/s. Angle commands are clipped to ±1.066 rad.
CommonRoad's `steering_constraints` clips rate to ±0.4 rad/s and blocks outward
motion at angle limits. CommonRoad applies its constraints again in the dynamics.
A 0.001 m/s denominator floor is used in the controller; tracking simulation
supports positive forward speeds only.

`src/tracking.py` reuses the KS dynamics imported from `src/vehicle.py` without
changing Milestone 1. The initial rear axle is `(0, 0.5)` m with yaw aligned to
the path tangent at x=0, zero steering, and speed 5 m/s. Acceleration input is
zero. Integration uses `solve_ivp`, `rtol=1e-9`, `atol=1e-11`, and maximum step
0.05 s. The controller is evaluated continuously inside the ODE. Simulation
stops at 20 seconds or when the front axle crosses the endpoint's normal plane.
Uniform output samples at approximately 50 Hz include the exact final time.

The returned `TrackingResult` records time, all five vehicle states, desired
steering angles, applied steering rates, CTE, heading errors, and termination
reason. MSE is the sample mean of front-axle `e_CTE²`; maximum error includes
the initial offset. Limit diagnostics inspect recorded commands and states.

Default results: MSE approximately **0.00803 m²**, maximum absolute CTE
**0.48734 m**. The initial correction reaches the steering-rate limit; neither
desired nor actual steering reaches the angle limits. Plots are saved to:

- `results/tracking_trajectory.png` (reference, rear axle, and front axle)
- `results/tracking_error.png` (front-axle CTE versus time)
- `results/tracking_steering.png` (actual and desired angles versus time)

The front axle tracks the reference; the rear axle follows a different curve.
This is a deterministic ideal kinematic experiment, without slip, noise,
sensor delays, or sampled-controller timing. The actuator is a fixed first-order
approximation. Nearest-point selection is intended for this gentle, nonintersecting
path; arbitrary looping paths would require progress-aware association. Metrics
and saturation diagnostics use output samples, so extremely brief between-sample
limit contacts could be missed. No gain optimization is performed.
