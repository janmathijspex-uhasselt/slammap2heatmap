"""Where to put the next scan station.

Turning the diagnostic into an instruction.  The heatmap says where the map
is weak; this says where to stand to fix it.

The trick is not to invent a model of what a new station would achieve.  The
map already contains the answer: it has six to eighteen stations, and the
validation pass measures exactly how much score they deliver as a function of
distance.  A new station is assumed to behave like the ones already there.
That decay curve is measured per map, so a sparse scan predicts sparse help.

Candidates are then ranked by how much of the *missing* score they would
recover, greedily, so the second suggestion does not simply restate the
first from a metre away.

Two honest limits, both reported with the results:

* Only cells known to be open floor can be candidates or beneficiaries.  The
  places that most need coverage are often the unscanned grey ones, and
  nothing here can see into those.
* The estimate ignores that a new station also creates landmarks on surfaces
  no current station can see.  Real improvement should therefore be at least
  this good, not worse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .config import DEFAULT, Config
from .grid import Grid
from .mapmodel import MapModel
from .scorers import Result
from .visibility import Visibility

# Thresholds live in config.py; see the `suggest` section there.

@dataclass
class Suggestion:
    rank: int
    xy: tuple[float, float]
    cell: int
    gain: float
    cells_helped: int
    mean_before: float
    mean_after: float

    def describe(self) -> str:
        return (
            f"  {self.rank}. x={self.xy[0]:7.2f}  y={self.xy[1]:7.2f}   "
            f"lifts {self.cells_helped:5d} cells "
            f"from {self.mean_before:.2f} to {self.mean_after:.2f} "
            f"(recovers {self.gain:.0f} score-cells)"
        )


@dataclass
class Plan:
    suggestions: list[Suggestion]
    decay: np.ndarray            # effectiveness at each distance step
    step: float
    coverage_before: float
    coverage_after: float
    target: float
    aggregate: str
    reachable: float

    def describe(self) -> str:
        if not self.suggestions:
            return "  nothing worth adding: the mapped area is already covered"
        lines = [
            "  measured station effectiveness: "
            + ", ".join(
                f"{i * self.step:.1f}m {v:.2f}"
                for i, v in enumerate(self.decay)
                if i % 4 == 0 and v > 0.01
            ),
            "  next scan positions, best first:",
        ]
        lines += [s.describe() for s in self.suggestions]
        lines.append(
            f"  together they would raise the share of open floor scoring above "
            f"{self.target:.2f} (best heading) from {100 * self.coverage_before:.0f}% "
            f"to {100 * self.coverage_after:.0f}%"
        )
        if self.reachable < self.target:
            lines.append(
                f"  but note: even standing at an existing station this map only "
                f"reaches {self.reachable:.2f}, so no number of new stations at "
                f"this scan quality will clear {self.target:.2f}. The limit is how "
                f"each station is captured, not how many there are."
            )
        lines.append(
            "  lower bound: a new station also creates landmarks on surfaces no "
            "current one can see, which this cannot predict"
        )
        return "\n".join(lines)


def _collapse(result: Result, aggregate: str) -> np.ndarray:
    if aggregate == "max":
        return result.combined.max(axis=1).astype(np.float64)
    if aggregate == "mean":
        return result.combined.mean(axis=1).astype(np.float64)
    if aggregate == "min":
        return result.combined.min(axis=1).astype(np.float64)
    raise ValueError(f"unknown aggregate {aggregate!r}")


def _measure_decay(model: MapModel, grid: Grid, result: Result,
                   reach: float, step: float, aggregate: str) -> np.ndarray:
    """How much score an existing station delivers, by distance. Measured, not assumed."""
    ix, iy = np.divmod(result.cells, grid.ny)
    xy = np.column_stack([
        grid.origin[0] + (ix + 0.5) * grid.res,
        grid.origin[1] + (iy + 0.5) * grid.res,
    ])
    d, _ = cKDTree(model.ground(model.stations.center)).query(xy)
    score = _collapse(result, aggregate)

    n = int(np.ceil(reach / step)) + 1
    curve = np.zeros(n)
    for i in range(n):
        sel = (d >= i * step) & (d < (i + 1) * step)
        curve[i] = score[sel].mean() if sel.sum() >= 3 else np.nan

    # Fill gaps, then force it to be non-increasing: a station cannot help
    # more further away, and sampling noise must not be read as if it could.
    idx = np.arange(n)
    good = ~np.isnan(curve)
    if not good.any():
        return np.zeros(n)
    curve = np.interp(idx, idx[good], curve[good])
    return np.minimum.accumulate(curve)


def recommend(model: MapModel, grid: Grid, vis: Visibility, result: Result,
              n: int = 5, cfg: Config | None = None,
              reach: float | None = None, target: float | None = None,
              aggregate: str | None = None) -> Plan:
    """Rank places to put the next scan stations."""
    c = (cfg or DEFAULT).suggest
    reach = c.reach_m if reach is None else reach
    step = c.decay_step_m
    target = c.target_score if target is None else target
    aggregate = c.aggregate if aggregate is None else aggregate
    decay = _measure_decay(model, grid, result, reach, step, aggregate)

    ix, iy = np.divmod(result.cells, grid.ny)
    xy = np.column_stack([
        grid.origin[0] + (ix + 0.5) * grid.res,
        grid.origin[1] + (iy + 0.5) * grid.res,
    ])
    current = _collapse(result, aggregate).copy()
    before = current.copy()

    tree = cKDTree(xy)
    neighbours = tree.query_ball_point(xy, reach, return_sorted=False)

    # Precompute, for every candidate, who it could reach and how much it
    # would give them.  Sightlines are already known from the sweep.
    reach_idx: list[np.ndarray] = []
    reach_val: list[np.ndarray] = []
    for c, nb in enumerate(neighbours):
        nb = np.asarray(nb, dtype=np.int64)
        delta = xy[nb] - xy[c]
        dist = np.hypot(delta[:, 0], delta[:, 1])
        bearing = np.arctan2(delta[:, 1], delta[:, 0])
        visible = dist <= vis.free_range[c][vis.bearing_bin(bearing)] + grid.res
        nb, dist = nb[visible], dist[visible]
        reach_idx.append(nb)
        reach_val.append(decay[np.minimum((dist / step).astype(int), len(decay) - 1)])

    chosen: list[Suggestion] = []
    for rank in range(1, n + 1):
        best_c, best_gain = -1, 0.0
        for c in range(len(xy)):
            if len(reach_idx[c]) == 0:
                continue
            gain = np.maximum(reach_val[c] - current[reach_idx[c]], 0.0).sum()
            if gain > best_gain:
                best_c, best_gain = c, gain
        if best_c < 0 or best_gain <= 0:
            break

        idx, val = reach_idx[best_c], reach_val[best_c]
        helped = val > current[idx]
        chosen.append(Suggestion(
            rank=rank,
            xy=(float(xy[best_c, 0]), float(xy[best_c, 1])),
            cell=int(result.cells[best_c]),
            gain=float(best_gain),
            cells_helped=int(helped.sum()),
            mean_before=float(current[idx[helped]].mean()) if helped.any() else 0.0,
            mean_after=float(val[helped].mean()) if helped.any() else 0.0,
        ))
        current[idx] = np.maximum(current[idx], val)

    return Plan(
        suggestions=chosen, decay=decay, step=step,
        coverage_before=float((before >= target).mean()),
        coverage_after=float((current >= target).mean()),
        target=target, aggregate=aggregate,
        reachable=float(decay[0]),
    )
