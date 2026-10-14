"""How far you can see from each cell, in every direction.

Testing occlusion by casting a ray per cell per landmark would be hopeless:
tens of thousands of cells against tens of thousands of landmarks.  But a
cell's sightlines do not depend on what it is looking at -- only on where the
walls are.  So sweep once per cell, recording the distance at which each
bearing first meets something solid, and every later occlusion test becomes a
lookup: a landmark at bearing b and distance d is hidden exactly when d
exceeds the free range at b.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT, Config
from .grid import Grid
from .occupancy import OCCUPIED, Occupancy



@dataclass
class Visibility:
    grid: Grid
    cells: np.ndarray         # (C,) flat grid indices that were swept
    free_range: np.ndarray    # (C, n_bearings) float32, metres
    n_bearings: int
    max_range: float

    def bearing_bin(self, bearing_rad: np.ndarray) -> np.ndarray:
        """Radians -> bearing bin index."""
        frac = (np.asarray(bearing_rad) / (2 * np.pi)) % 1.0
        return np.minimum((frac * self.n_bearings).astype(np.int64),
                          self.n_bearings - 1)

    def subset(self, mask: np.ndarray) -> "Visibility":
        """The same sweep restricted to some of its cells."""
        return Visibility(
            grid=self.grid, cells=self.cells[mask],
            free_range=self.free_range[mask],
            n_bearings=self.n_bearings, max_range=self.max_range,
        )

    def summary(self) -> str:
        med = float(np.median(self.free_range))
        return (
            f"{len(self.cells)} cells swept, {self.n_bearings} bearings, "
            f"median free range {med:.1f} m"
        )


def sweep(occ: Occupancy, cells: np.ndarray,
          cfg: Config | None = None) -> Visibility:
    """March outward from every given cell until each bearing hits a wall."""
    c = (cfg or DEFAULT).visibility
    n_bearings, max_range = c.n_bearings, c.max_range_m
    grid = occ.grid
    solid = (occ.state == OCCUPIED).ravel()

    ix, iy = np.divmod(cells, grid.ny)
    cx = grid.origin[0] + (ix + 0.5) * grid.res
    cy = grid.origin[1] + (iy + 0.5) * grid.res

    free_range = np.full((len(cells), n_bearings), max_range, dtype=np.float32)
    step = grid.res * 0.7
    radii = np.arange(grid.res, max_range, step, dtype=np.float32)

    for b in range(n_bearings):
        theta = 2 * np.pi * (b + 0.5) / n_bearings
        dx, dy = np.cos(theta), np.sin(theta)
        alive = np.ones(len(cells), dtype=bool)
        for r in radii:
            live = np.flatnonzero(alive)
            if live.size == 0:
                break
            px = cx[live] + dx * r
            py = cy[live] + dy * r
            flat = grid.sample_flat(np.column_stack([px, py]))
            # Off the grid counts as blocked: there is nothing mapped out there.
            blocked = (flat < 0) | solid[np.maximum(flat, 0)]
            hit = live[blocked]
            if hit.size:
                free_range[hit, b] = r
                alive[hit] = False

    return Visibility(grid=grid, cells=cells, free_range=free_range,
                      n_bearings=n_bearings, max_range=float(max_range))
