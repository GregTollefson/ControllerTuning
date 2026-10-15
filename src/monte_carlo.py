"""Monte Carlo evaluation of a fixed Stanley gain on a common path segment."""
import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from src.path import ReferencePath
from src.scenarios import generate_scenarios
from src.tracking import simulate_tracking


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: int
    speed: float
    offset: float
    k: float
    endpoint: float
    mse: float | None
    max_error: float | None
    angle_saturated: bool | None
    rate_saturated: bool | None
    status: str  # completed, incomplete, failed
    elapsed_simulation: float | None
    message: str = ""


@dataclass(frozen=True)
class Evaluation:
    results: tuple[ScenarioResult, ...]
    statistics: dict
    execution_seconds: float


def summarize(results):
    """Summaries condition on completion; failed/incomplete counts stay explicit."""
    completed = [r for r in results if r.status == "completed"]
    losses = np.array([r.mse for r in completed], dtype=float)
    n = len(completed)
    mean = float(np.mean(losses)) if n else None
    sd = float(np.std(losses, ddof=1)) if n > 1 else None
    return {
        "generated": len(results),
        "completed": n,
        "incomplete": sum(r.status == "incomplete" for r in results),
        "failed": sum(r.status == "failed" for r in results),
        "mean_mse": mean,
        "sample_sd_mse": sd,
        "mc_standard_error": sd / np.sqrt(n) if sd is not None else None,
        "tracking_rmse": float(np.sqrt(mean)) if mean is not None else None,
        "max_abs_cte": max((r.max_error for r in completed), default=None),
        # Saturation counts cover all runs with available diagnostics.
        "angle_saturated_scenarios": sum(r.angle_saturated is True for r in results),
        "rate_saturated_scenarios": sum(r.rate_saturated is True for r in results),
        "unknown_limit_scenarios": sum(r.angle_saturated is None for r in results),
    }


def evaluate_gain(k, scenarios, endpoint=100.0, max_time=40.0):
    """Evaluate common random scenarios without resampling or changing controller law.

    Completed-run mean is a conditional loss, not a failure-penalized objective.
    Callers must inspect completion counts before comparing gains.
    """
    scenarios = tuple(scenarios)
    if not scenarios or len({s.scenario_id for s in scenarios}) != len(scenarios):
        raise ValueError("Provide a nonempty scenario set with unique IDs")
    path = ReferencePath()  # identical geometry, independent of speed/gain
    if not np.isfinite(k) or k < 0 or not np.isfinite(max_time) or max_time <= 0:
        raise ValueError("Finite nonnegative gain and positive maximum time required")
    if not np.isfinite(endpoint) or not 3 < endpoint <= path.length:
        raise ValueError("Endpoint must be beyond initial front axle and within path")
    start = perf_counter()
    results = []
    for scenario in scenarios:
        inputs = dict(scenario_id=scenario.scenario_id, speed=scenario.speed,
                      offset=scenario.offset, k=float(k), endpoint=float(endpoint))
        try:
            run = simulate_tracking(path=path, k=k, speed=scenario.speed,
                                    offset=scenario.offset, offset_frame="path_normal",
                                    endpoint=endpoint, duration=max_time)
            limits = run.limits_reached
            completed = run.termination == "path_end"
            results.append(ScenarioResult(
                **inputs, mse=run.mse, max_error=run.max_error,
                angle_saturated=limits["angle"], rate_saturated=limits["rate"],
                status="completed" if completed else "incomplete",
                elapsed_simulation=float(run.time[-1]),
                message="" if completed else "Maximum simulation time reached before endpoint"))
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            results.append(ScenarioResult(
                **inputs, mse=None, max_error=None, angle_saturated=None,
                rate_saturated=None, status="failed", elapsed_simulation=None,
                message=f"{type(error).__name__}: {error}"))
    results = tuple(results)
    return Evaluation(results, summarize(results), perf_counter() - start)


def save_outputs(evaluation, output_dir):
    """CSV retains every attempted scenario; plots show only completed losses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in evaluation.results]
    with (output_dir / "monte_carlo_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(ScenarioResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)
    completed = [r for r in evaluation.results if r.status == "completed"]
    losses = [r.mse for r in completed]
    counts = evaluation.statistics
    title = (f"Completed {counts['completed']}/{counts['generated']}; "
             f"incomplete {counts['incomplete']}; failed {counts['failed']}")
    for name, xlabel, attribute in [
        ("monte_carlo_mse_histogram", "Scenario tracking MSE [m²]", None),
        ("monte_carlo_mse_vs_speed", "Speed [m/s]", "speed"),
        ("monte_carlo_mse_vs_offset", "Initial left-normal offset [m]", "offset"),
    ]:
        fig, ax = plt.subplots()
        if attribute is None:
            ax.hist(losses, bins="auto" if losses else 1, edgecolor="black")
            ax.set_ylabel("Completed scenarios")
        else:
            ax.scatter([getattr(r, attribute) for r in completed], losses)
            ax.set_ylabel("Scenario tracking MSE [m²]")
        ax.set(xlabel=xlabel, title=title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--endpoint", type=float, default=100.0)
    parser.add_argument("--max-time", type=float, default=40.0)
    args = parser.parse_args()
    start = perf_counter()
    evaluation = evaluate_gain(args.gain, generate_scenarios(args.count, args.seed),
                               endpoint=args.endpoint, max_time=args.max_time)
    save_outputs(evaluation, Path(__file__).resolve().parents[1] / "results")
    for key, value in evaluation.statistics.items():
        print(f"{key}: {value}")
    print(f"Simulation evaluation time: {evaluation.execution_seconds:.3f} s")
    print(f"Total experiment time (including plots and CSV): {perf_counter() - start:.3f} s")


if __name__ == "__main__":
    main()
