"""Smooth centerline y=A sin(2*pi*x/wavelength), traversed in increasing x.

World x points forward and y left; headings are counterclockwise radians.
Signed CTE is positive when the reference lies left of the query point.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class ReferencePath:
    length: float = 120.0  # x extent, not arc length [m]
    amplitude: float = 3.0
    wavelength: float = 60.0

    def __post_init__(self):
        if not np.all(np.isfinite([self.length, self.amplitude, self.wavelength])):
            raise ValueError("Path parameters must be finite")
        if self.length <= 0 or self.wavelength <= 0:
            raise ValueError("Length and wavelength must be positive")

    def y(self, x):
        return self.amplitude * np.sin(2 * np.pi * x / self.wavelength)

    def heading(self, x):
        return np.arctan(self.amplitude * 2 * np.pi / self.wavelength
                         * np.cos(2 * np.pi * x / self.wavelength))

    def nearby_point(self, x, y):
        """Project onto the finite path using a coarse search and local refinement."""
        grid = np.linspace(0, self.length, 601)
        distance2 = lambda s: (s - x)**2 + (self.y(s) - y)**2
        index = np.argmin(distance2(grid))
        lo, hi = grid[max(0, index - 1)], grid[min(len(grid) - 1, index + 1)]
        result = minimize_scalar(distance2, bounds=(lo, hi), method="bounded",
                                 options={"xatol": 1e-10})
        s = min([lo, result.x, hi], key=distance2)
        return float(s), float(self.y(s))

    def tracking_error(self, x, y):
        """Return reference heading and signed normal error at nearest point."""
        rx, ry = self.nearby_point(x, y)
        heading = self.heading(rx)
        error = -np.sin(heading) * (rx - x) + np.cos(heading) * (ry - y)
        return float(heading), float(error)
