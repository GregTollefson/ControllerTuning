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

# Milestone 3: Monte Carlo driving scenarios

Run the default experiment and the full test suite in the existing environment:

```bash
MPLCONFIGDIR=/tmp/controller-tuning-mpl .venv/bin/python -m src.monte_carlo
.venv/bin/python -m unittest discover -s tests -v
```

Optional CLI arguments are `--count`, `--seed`, `--gain`, `--endpoint`, and
`--max-time`. Defaults are 20 scenarios, seed 123, Stanley gain 1.0 /s,
endpoint x=100 m, and a 40 s maximum simulated time per scenario. No packages
or controller equations changed, and the actuator gain remains 5.0 /s.

`src/scenarios.py` uses NumPy `default_rng(seed)` to independently sample
speed from Uniform(4, 8) m/s and initial signed lateral offset from
Uniform(-0.5, 0.5) m. Each immutable scenario has a unique integer ID.
Scenario generation is separate from evaluation, so the same sampled inputs
can be reused across gains (common random numbers):

```python
from src.scenarios import generate_scenarios
from src.monte_carlo import evaluate_gain

scenarios = generate_scenarios(count=20, seed=123)
baseline = evaluate_gain(1.0, scenarios)
alternative = evaluate_gain(0.8, scenarios)
print(baseline.statistics)
```

All Monte Carlo runs use the unchanged reference `y=3 sin(2πx/60)` on
0 <= x <= 120 m. They start from the same reference station x=0, with rear
axle displaced by `offset * [-sin(heading(0)), cos(heading(0))]`. Positive
sampled offset means left of the path; Stanley's front-axle CTE uses the
opposite corrective sign. Yaw is initially the tangent heading at x=0,
steering is zero, and speed remains constant through zero acceleration.
The sampled rear-axle offset is not exactly the initial front-axle CTE on a
curved path.

`simulate_tracking` now accepts `k=1.0`, `speed`, `offset`, `endpoint`, and
`offset_frame`. Its default `offset_frame="world_y"`, offset 0.5 m, and
20 s duration preserve the Milestone 2 demonstration exactly. Monte Carlo
explicitly selects `offset_frame="path_normal"` as defined above.

The shared target is the reference station **x=100 m**, measured in world x,
not arc length or distance traveled. Every run terminates when its front axle
crosses that station's normal plane in the forward direction. The reference
continues past the target, so nearest-point association is not clipped at
this stopping boundary. Actual front-axle starting x varies slightly with
normal offset, but the starting reference station and target station are
identical for all scenarios. A 40 s safeguard prevents indefinitely long
runs; integration and output sampling retain Milestone 2 tolerances.

For scenario i with N_i uniformly spaced output samples (about 50 Hz,
including the exact terminal time), the loss and Monte Carlo estimator are:

```text
L_i(k) = (1/N_i) sum_j e_front,i(t_j; k)^2
J_hat(k) = (1/n) sum_i L_i(k)
s = sqrt(sum_i (L_i(k) - J_hat(k))^2 / (n - 1))
MCSE = s / sqrt(n)
tracking_RMSE = sqrt(J_hat(k))
```

Here n is the number of completed scenarios (n=M when all complete).
Each scenario has equal weight despite differing traversal times/sample counts.
The sample standard deviation s (`ddof=1`) measures variability of scenario
losses; the Monte Carlo standard error estimates uncertainty in their mean.
It does not measure individual tracking error or variability across time.
SD and MCSE are unavailable (`None`) with fewer than two completed runs; all
loss summaries are unavailable when none complete.

`evaluate_gain` returns individual `ScenarioResult` records, aggregate
statistics, and wall-clock evaluation time. Status is `completed` only for
endpoint termination, `incomplete` for the time safeguard, or `failed` for a
caught validation/integration error. Incomplete runs retain their partial
metrics; failures retain an error message and unavailable metrics/limit
indicators. Every input scenario remains represented. Completed-only
statistics are explicitly conditional on completion; completion/failure
counts must be considered before comparing gains. No arbitrary failure
penalty or Bayesian optimization is implemented.

Default validation results (`k=1.0`, M=20, seed=123):

| Quantity | Result |
| --- | ---: |
| Generated / completed scenarios | 20 / 20 |
| Incomplete / failed scenarios | 0 / 0 |
| Mean scenario MSE | 0.00430127549 m² |
| Tracking RMSE | 0.0655841100 m |
| Sample SD of scenario MSE | 0.00283576123 m² |
| Monte Carlo standard error | 0.000634095489 m² |
| Maximum absolute front-axle CTE | 0.476927877 m |
| Scenarios reaching steering-angle limits | 0 |
| Scenarios reaching steering-rate limits | 3 |
| Evaluation wall time | 5.561 s |
| Total experiment wall time, including plots and CSV | 5.990 s |

Times are machine-dependent. Maximum CTE is over completed runs; saturation
counts cover all runs with diagnostics, including incomplete runs. The unknown
limit count identifies failures without diagnostics. Counts refer to scenarios
that touched a limit, not numbers of individual saturation events. Angle
saturation includes either a clipped desired command or an actual angle at
the bound; rate saturation uses recorded constrained steering-rate inputs.

Outputs (previous milestone plots are preserved):

- `results/monte_carlo_results.csv`: every scenario's inputs, gain, endpoint,
  loss, maximum CTE, completion status, simulated duration, limit indicators,
  and any failure message. Unavailable values are blank.
- `results/monte_carlo_mse_histogram.png`
- `results/monte_carlo_mse_vs_speed.png`
- `results/monte_carlo_mse_vs_offset.png`

Plots explicitly state completion, incomplete, and failure counts and display
completed-run losses. The full suite also checks seeded reproducibility,
parameter bounds, reuse across gains, hand-calculated statistical summaries,
normal-offset geometry, endpoint termination, and explicit failure accounting.
Tests use short segments and small batches to remain fast.

Limitations from Milestone 2 still apply: no tire slip, traffic, sensors, or
stochastic disturbances within a run. Randomness is only in the initial
conditions. Sample-based losses and saturation flags can miss very brief
between-sample behavior. Endpoint-plane crossing is suitable for this gentle,
nonintersecting path; it is not an off-road or lane-departure safety criterion.
The 20-scenario estimate is a small-sample baseline, and completion-conditioned
means alone must not be treated as a valid objective for gains that fail runs.

# Milestone 4: Gaussian Process Bayesian Optimization

Reproduce the full experiment and run all tests with the existing environment:

```bash
MPLCONFIGDIR=/tmp/controller-tuning-mpl .venv/bin/python -m src.optimization
.venv/bin/python -m unittest discover -s tests -v
```

The installed scikit-learn supplies `GaussianProcessRegressor`; no Bayesian
optimization framework or additional package is installed. The CommonRoad
model, Stanley equations, actuator gain, reference path, and Milestone 1–3
outputs are unchanged.

`src/objective.py` defines `TrackingObjective(scenarios)`, a reusable callable
of gain k with fixed scenarios and simulation settings. Its scalar loss is the
mean of **all** scenario losses. A completed run contributes its front-axle MSE;
an incomplete, failed, or invalid-loss run contributes **1.0 m²** (equivalent
to 1 m RMS tracking error, far above the nominal losses in this experiment).
The penalty is finite, configurable, and does not cap valid completed-run
losses. It is a modeling choice rather than a physical failure cost. Completion,
incompletion, failure, and saturation diagnostics remain separately available
through `diagnostics(k)` and the scenario-level CSVs. Outputs are always finite
for valid gain inputs; invalid gains are rejected. Exact repeated floating-point
gains are cached within each objective instance. Failed evaluations are cached
too, because inputs and dynamics are deterministic.

`src/gp_bo.py` fits a covariance amplitude times a Matérn kernel with ν=2.5.
Gains are linearly mapped from [0.1, 3.0] to [0, 1]; target losses are normalized
by their training mean and standard deviation (`normalize_y=True`). The
covariance amplitude is optimized within [0.001, 1000] and length scale within
[0.01, 10] in scaled input units. Hyperparameters use scikit-learn's default
L-BFGS-B optimizer, two additional restarts, and seed 321. `alpha=1e-8` adds
small diagonal numerical regularization in normalized loss units, rather than
modeling independent Monte Carlo measurement noise: the fixed-scenario loss is
effectively deterministic. Prediction exposes mean and standard deviation in
original MSE units. API reference:
[scikit-learn GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html).

Expected Improvement (EI) is computed for minimization using SciPy normal
CDF Φ and PDF φ:

```text
I(k) = best_observed_loss - mu(k) - xi
Z(k) = I(k) / sigma(k)
EI(k) = I(k) Phi(Z(k)) + sigma(k) phi(Z(k))
```

Exploration `xi=1e-6 m²` is configurable and nonnegative. EI is set to zero
when predictive SD <= 1e-12; negative numerical artifacts are clipped to zero,
and extreme floating-point outputs are made finite. Each iteration maximizes
EI over 4001 equally spaced candidate gains. Candidates within 1e-6 /s of a
previous observation are excluded; exact EI ties prefer larger uncertainty.
After the 25th observation the GP is refitted, and the final posterior and EI
are plotted without requesting another evaluation.

Both methods use five unique initial gains `[0.1, 0.825, 1.55, 2.275, 3.0]`.
Each has a total budget of **25 gain evaluations**, including those five.
GP-BO selects 20 more using EI. Random search samples 20 more uniformly from
[0.1, 3.0] using `default_rng(789)` (resampling only numerical duplicates).
Both use the exact same 20 training scenarios from seed 123, the same loss
function, x=100 m endpoint, and 40 s maximum simulated time. Cached shared
initial results still count as five observations for each method. The fixed
k=1 baseline training evaluation is additional reporting work, outside both
search budgets, and cannot influence gain selection.

The best **observed training** gain from each method is frozen before evaluating
an independent set of 100 scenarios generated with seed 456. The fixed baseline
and both selected controllers use the same held-out scenarios, allowing paired
comparisons. Held-out data are not used for acquisition, selection, or retuning.
If both methods select the same gain, its held-out batch is cached and reused.
The held-out CSV reports the penalized objective, its square-root RMSE, sample
SD (`ddof=1`) and Monte Carlo SE of all scenario losses, completion counts, and
counts of scenarios reaching angle/rate limits. Thus failures cannot disappear
from held-out metrics either. Completed-only statistics remain available in
the underlying Monte Carlo diagnostic objects.

New artifacts in `results/`:

- `gp_bo_history.csv` and `random_search_history.csv`: method, evaluation count,
  gain, objective, and cumulative best objective.
- `heldout_comparison.csv`: baseline and selected-controller comparisons.
- `optimization_training_scenarios.csv` and `optimization_heldout_scenarios.csv`:
  every simulated scenario, gain, outcome, and available diagnostics.
- `optimization_summary.json`: settings, numerical results, final kernel,
  captured GP warning messages, physical simulation count, and wall time.
- `gp_bo_surrogate.png`: final GP mean and 95% pointwise surrogate band.
- `gp_bo_expected_improvement.png`: EI conditioned on all 25 GP-BO observations.
- `gp_bo_best_so_far.png`: equal-budget convergence comparison.
- `gp_bo_heldout_comparison.png`: independent held-out objective comparison.

The GP band is `mu ± 1.96 sigma`: it describes uncertainty of the fitted
surrogate under its covariance model. It is **not** a confidence interval for
the population expected tracking loss or a Monte Carlo uncertainty estimate.
MCSE describes finite-scenario uncertainty in the estimated mean instead.
A single training seed and random-search seed cannot establish superiority
of one search strategy. Common scenarios correlate estimates across gains,
and finite training sets can cause selection bias. The interval restricts
which gains can be selected; a boundary solution does not establish a global
optimum. Penalty transitions can make the objective less smooth than the
Matérn surrogate assumes. Existing kinematic-model and sampled-diagnostic
limitations still apply. Tests use synthetic functions and mocked scenario
failures rather than repeating the expensive optimization experiment.

Observed validation results (20 training scenarios; 100 held-out scenarios):

| Controller | Gain k [1/s] | Training MSE [m²] | Held-out MSE [m²] | Held-out RMSE [m] | Scenario-loss SD [m²] | MCSE [m²] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed baseline | 1.0 | 0.00430127549 | 0.00354016225 | 0.0594992626 | 0.00237638224 | 0.000237638224 |
| GP-BO selection | 3.0 | 0.00179885103 | 0.00149465656 | 0.0386607883 | 0.00137254709 | 0.000137254709 |
| Random-search selection | 3.0 | 0.00179885103 | 0.00149465656 | 0.0386607883 | 0.00137254709 | 0.000137254709 |

All three held-out entries completed 100/100 scenarios, with zero incomplete
or failed runs and zero steering-angle-saturated scenarios. Steering rate
saturated in 13/100 baseline scenarios and 65/100 scenarios for each selected
controller. The two selected entries are the same controller evaluated on the
same held-out inputs, not independent replications.

Both searches attained their final best value in the shared initial design
at evaluation 5; neither improved it in the remaining 20 evaluations. Their
best-so-far curves therefore overlap. This experiment shows lower observed
held-out tracking loss for k=3 than for k=1, with more rate saturation. It
provides no evidence that GP-BO outperforms random search, nor that k=3 is
optimal outside the prescribed interval.

The experiment performed **1120 actual vehicle simulations**: 45 distinct
search gains × 20 scenarios, one additional baseline batch × 20, and two
unique held-out gains × 100. Shared initial evaluations and identical selected
held-out gains were cached; both methods still received exactly 25 gain
observations. All 920 training and 200 distinct held-out simulations completed.
Measured total experiment wall time was **320.978 s** (about 5.35 minutes),
including GP fitting, CSVs, and plots; times depend on the machine. The full
suite passed **26 tests** in 1.656 s; test simulations are outside the reported
experiment count and runtime.

GP fitting emitted **26 L-BFGS abnormal line-search termination convergence
warnings** across hyperparameter restarts/refits. They are retained verbatim
in `optimization_summary.json`; they were not silently discarded. Inputs and
targets were already scaled. Fitting returned models and predictions remained
finite; no vehicle integrations failed. The final fitted kernel was approximately
`28.5**2 * Matern(length_scale=1.92, nu=2.5)` in scaled coordinates. These warnings
mean numerical hyperparameter convergence is not assured; the surrogate band
and acquisition choices should be interpreted with that limitation. No settings
were retuned using the held-out results.
