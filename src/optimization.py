"""Reproduce the equal-budget GP-BO/random-search and held-out experiment."""
import csv
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.gp_bo import BOUNDS, expected_improvement, optimize_bo, random_search
from src.objective import TrackingObjective, penalized_losses
from src.scenarios import generate_scenarios


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(output, bo, random, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.linspace(*BOUNDS, 4001)
    mean, std = bo.gp.predict(grid)
    fig, ax = plt.subplots()
    ax.plot(grid, mean, label="GP mean")
    ax.fill_between(grid, mean - 1.96*std, mean + 1.96*std, alpha=0.25,
                    label="95% pointwise surrogate band")
    ax.scatter([r.gain for r in bo.history], [r.objective for r in bo.history],
               s=18, color="black", label="Evaluated training losses")
    ax.set(xlabel="Stanley gain k [1/s]", ylabel="Training objective [m²]",
           title="Final GP surrogate (not Monte Carlo confidence)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "gp_bo_surrogate.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(grid, expected_improvement(mean, std, bo.best.objective))
    ax.set(xlabel="Stanley gain k [1/s]", ylabel="Expected Improvement [m²]",
           title="EI after all 25 GP-BO observations")
    fig.tight_layout()
    fig.savefig(output / "gp_bo_expected_improvement.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    for search, label in [(bo, "GP-BO"), (random, "Random search")]:
        ax.plot([r.evaluation for r in search.history],
                [r.cumulative_best for r in search.history], label=label)
    ax.set(xlabel="Gain evaluations (including shared initial five)",
           ylabel="Best training objective [m²]", yscale="log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "gp_bo_best_so_far.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar([r["controller"] for r in comparison], [r["heldout_objective"] for r in comparison])
    ax.set(ylabel="Held-out objective MSE [m²]", title="Independent seed 456: 100 shared scenarios")
    fig.tight_layout()
    fig.savefig(output / "gp_bo_heldout_comparison.png", dpi=150)
    plt.close(fig)


def main():
    start = perf_counter()
    output = Path(__file__).resolve().parents[1] / "results"
    output.mkdir(exist_ok=True)
    training = TrackingObjective(generate_scenarios(20, seed=123))

    def logged_objective(k):
        value = training(k)
        print(f"Training k={k:.8f}: objective={value:.10g}; "
              f"unique batches={len(training.cache)}", flush=True)
        return value

    bo = optimize_bo(logged_objective)
    random = random_search(logged_objective)
    write_csv(output / "gp_bo_history.csv", [asdict(r) for r in bo.history])
    write_csv(output / "random_search_history.csv", [asdict(r) for r in random.history])
    # Freeze selections before constructing or examining any held-out evaluations.
    selected = [("Fixed baseline", 1.0), ("GP-BO", bo.best.gain),
                ("Random search", random.best.gain)]
    heldout = TrackingObjective(generate_scenarios(100, seed=456))
    comparison = []
    for label, gain in selected:
        train_value = training(gain)
        heldout_value = heldout(gain)
        evaluation = heldout.diagnostics(gain)
        losses = penalized_losses(evaluation)
        sd = float(np.std(losses, ddof=1))
        stats = evaluation.statistics
        row = dict(controller=label, gain=gain, training_objective=train_value,
                   heldout_objective=heldout_value, heldout_rmse=float(np.sqrt(heldout_value)),
                   scenario_loss_sd=sd, mc_standard_error=sd/np.sqrt(len(losses)),
                   completed=stats["completed"], incomplete=stats["incomplete"],
                   failed=stats["failed"], angle_saturated=stats["angle_saturated_scenarios"],
                   rate_saturated=stats["rate_saturated_scenarios"],
                   unknown_limits=stats["unknown_limit_scenarios"])
        comparison.append(row)
        print(json.dumps(row), flush=True)
    write_csv(output / "heldout_comparison.csv", comparison)
    # Preserve scenario-level diagnostic evidence, including all unsuccessful runs.
    for name, objective in [("optimization_training_scenarios", training),
                            ("optimization_heldout_scenarios", heldout)]:
        rows = [asdict(r) for _, evaluation in objective.cache.values() for r in evaluation.results]
        write_csv(output / f"{name}.csv", rows)
    save_plots(output, bo, random, comparison)
    summary = dict(
        settings=dict(bounds=BOUNDS, budget_per_method=25, initial_count=5,
                      training_count=20, training_seed=123, heldout_count=100,
                      heldout_seed=456, gp_seed=321, random_seed=789, xi=1e-6,
                      grid_size=4001, endpoint=100.0, max_time=40.0, failure_penalty=1.0),
        comparison=comparison, final_kernel=str(bo.gp.model.kernel_),
        gp_warnings=bo.gp.warnings, training_unique_batches=len(training.cache),
        heldout_unique_batches=len(heldout.cache),
        vehicle_simulations=training.vehicle_simulations + heldout.vehicle_simulations,
        runtime_seconds=perf_counter() - start)
    (output / "optimization_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
