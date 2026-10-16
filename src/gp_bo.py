"""One-dimensional Matérn Gaussian process and grid Expected Improvement search."""
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


BOUNDS = (0.1, 3.0)
DUPLICATE_TOLERANCE = 1e-6


def expected_improvement(mean, std, best, xi=1e-6):
    """Minimization EI in objective units; near-zero uncertainty has zero EI."""
    mean, std = np.broadcast_arrays(np.asarray(mean, float), np.asarray(std, float))
    if (not np.isfinite(mean).all() or not np.isfinite(std).all()
            or np.any(std < 0) or not np.isfinite(best) or not np.isfinite(xi) or xi < 0):
        raise ValueError("EI requires finite inputs, nonnegative std and xi")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        improvement = best - mean - xi
        safe_std = np.maximum(std, 1e-15)
        z = improvement / safe_std
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    ei = np.nan_to_num(ei, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)
    return np.maximum(np.where(std > 1e-12, ei, 0.0), 0.0)


class GaussianProcess:
    """Scale gains to [0,1]; normalize losses by training mean and SD."""
    def __init__(self, bounds=BOUNDS, seed=321):
        self.bounds = validate_settings(bounds, 5, 5)
        self.model = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=0.2, length_scale_bounds=(1e-2, 10.0), nu=2.5),
            alpha=1e-8, normalize_y=True, n_restarts_optimizer=2, random_state=seed)
        self.warnings = []

    def scale(self, gains):
        lo, hi = self.bounds
        return ((np.asarray(gains, float).reshape(-1, 1) - lo) / (hi - lo))

    def fit(self, gains, losses):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.model.fit(self.scale(gains), np.asarray(losses, float))
        self.warnings.extend(str(w.message) for w in caught)
        return self

    def predict(self, gains):
        return self.model.predict(self.scale(gains), return_std=True)


def validate_settings(bounds, budget, initial_count):
    lo, hi = bounds
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("Bounds must be finite and increasing")
    if (not isinstance(budget, int) or not isinstance(initial_count, int)
            or initial_count < 2 or budget < initial_count):
        raise ValueError("Integer budget must cover at least two initial gains")
    return float(lo), float(hi)


@dataclass(frozen=True)
class Observation:
    method: str
    evaluation: int
    gain: float
    objective: float
    cumulative_best: float


@dataclass
class SearchResult:
    history: list[Observation]
    gp: GaussianProcess | None = None

    @property
    def best(self):
        return min(self.history, key=lambda row: row.objective)


def observe(objective, gains, losses, history, gain, method):
    value = float(objective(float(gain)))
    if not np.isfinite(value):
        raise ValueError("Objective must return a finite scalar")
    gains.append(float(gain))
    losses.append(value)
    history.append(Observation(method, len(gains), float(gain), value, min(losses)))


def optimize_bo(objective, bounds=BOUNDS, budget=25, initial_count=5,
                seed=321, xi=1e-6, grid_size=4001):
    lo, hi = validate_settings(bounds, budget, initial_count)
    if grid_size < budget or not np.isfinite(xi) or xi < 0:
        raise ValueError("Grid must cover budget; exploration must be nonnegative")
    gains, losses, history = [], [], []
    for gain in np.linspace(lo, hi, initial_count):
        observe(objective, gains, losses, history, gain, "gp_bo")
    gp = GaussianProcess(bounds, seed)
    grid = np.linspace(lo, hi, grid_size)
    while len(history) < budget:
        gp.fit(gains, losses)
        mean, std = gp.predict(grid)
        ei = expected_improvement(mean, std, min(losses), xi)
        available = np.all(np.abs(grid[:, None] - np.array(gains)) > DUPLICATE_TOLERANCE, axis=1)
        if not available.any():
            raise ValueError("No unevaluated candidates remain")
        # Deterministic tie-breaking: among equal EI choose largest uncertainty.
        candidates = np.flatnonzero(available)
        order = np.lexsort((std[candidates], ei[candidates]))
        gain = grid[candidates[order[-1]]]
        observe(objective, gains, losses, history, gain, "gp_bo")
    gp.fit(gains, losses)  # posterior and EI plots condition on all observations
    return SearchResult(history, gp)


def random_search(objective, bounds=BOUNDS, budget=25, initial_count=5, seed=789):
    lo, hi = validate_settings(bounds, budget, initial_count)
    rng = np.random.default_rng(seed)
    gains, losses, history = [], [], []
    for gain in np.linspace(lo, hi, initial_count):
        observe(objective, gains, losses, history, gain, "random_search")
    while len(history) < budget:
        gain = rng.uniform(lo, hi)
        if np.all(np.abs(np.array(gains) - gain) > DUPLICATE_TOLERANCE):
            observe(objective, gains, losses, history, gain, "random_search")
    return SearchResult(history)
