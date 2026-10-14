"""Check the predictions against the one thing we can actually verify.

The scorers rest on constants -- how many matched points tracking needs, how
much of a viewpoint change ORB survives -- that cannot be read out of a map
file.  Left unchecked they are guesses, and a heatmap built on guesses is
decoration.

Three tests, in increasing order of how much they can tell us:

* **At its own keyframes.** Standing exactly where the scanner stood, facing
  exactly where it faced, is the one pose where localization is known to
  succeed.  The model must say so.  Views that saw almost nothing are excluded
  -- a scan pointed at a blank wall genuinely cannot be relocalized against.
* **Decay with distance.** Quality has to fall off away from the stations,
  monotonically.  Where it falls off says how far apart stations can be.
* **Leave one station out.** The real test.  Remove a station, rebuild the map
  without it, and ask the model to predict that spot blind.  Comparing that
  with what the full map measures there is as close to ground truth as a
  single map can give.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from . import loader, mapmodel, scorers
from .config import DEFAULT, Config
from .grid import Grid
from .mapmodel import MapModel
from .occupancy import Occupancy
from .scorers import Result
from .visibility import Visibility



@dataclass
class PoseCheck:
    n: int
    n_skipped: int
    track: np.ndarray
    reloc: np.ndarray
    combined: np.ndarray
    matches: np.ndarray

    @property
    def ok(self) -> bool:
        return (
            self.n > 0
            and float(np.median(self.track)) >= 0.8
            and float(np.median(self.reloc)) >= 0.8
        )

    def verdict(self) -> str:
        """What a failure here actually means.

        The test can fail two ways, and they call for opposite responses. If
        the *model* were wrong it would fail on every map, since the constants
        are shared. If one map fails while its siblings pass, the map is the
        problem -- and saying so is the whole point of a validator.
        """
        if self.n == 0:
            return "no view held enough landmarks to be a fair test"
        if self.ok:
            return "OK - the map localizes where it was built, as it must"
        weak = f"{self.n_skipped} of {self.n + self.n_skipped} views"
        if float(np.median(self.track)) >= 0.8:
            return (
                f"WEAK MAP - trackable, but it cannot be relocalized against "
                f"reliably even standing where it was scanned ({weak} were too "
                f"sparse to even test). Expect devices to fail to start up here."
            )
        return (
            f"WEAK MAP - too thin to localize against even where it was scanned "
            f"({weak} were too sparse to even test). This map needs rescanning, "
            f"not reprocessing."
        )

    def describe(self) -> str:
        if self.n == 0:
            return f"at own keyframes: {self.verdict()}"
        return (
            f"at own keyframes ({self.n} views, {self.n_skipped} too sparse to test): "
            f"track {np.median(self.track):.2f}, reloc {np.median(self.reloc):.2f}, "
            f"combined {np.median(self.combined):.2f}, "
            f"matches {np.median(self.matches):.0f}\n"
            f"      {self.verdict()}"
        )


@dataclass
class DecayBand:
    lo: float
    hi: float
    n: int
    track: float
    reloc: float
    combined: float


@dataclass
class Holdout:
    station: int
    n_cells: int
    with_station: float
    without_station: float

    @property
    def drop(self) -> float:
        return self.with_station - self.without_station


@dataclass
class Report:
    poses: PoseCheck
    decay: list[DecayBand]
    holdouts: list[Holdout] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"  {self.poses.describe()}"]
        mono = all(
            self.decay[i].combined >= self.decay[i + 1].combined - 1e-6
            for i in range(len(self.decay) - 1)
        )
        lines.append(
            f"  decay with distance from a station "
            f"({'monotonic, as it must be' if mono else 'NOT monotonic - suspicious'}):"
        )
        lines.append(f"      {'range':>9}  {'cells':>6}  {'track':>6} {'reloc':>6} {'comb':>6}")
        for d in self.decay:
            rng = f"{d.lo:.0f}-{d.hi:.0f} m" if np.isfinite(d.hi) else f"{d.lo:.0f}+ m"
            lines.append(
                f"      {rng:>9}  {d.n:6d}  {d.track:6.2f} {d.reloc:6.2f} {d.combined:6.2f}"
            )
        if self.holdouts:
            drops = np.array([h.drop for h in self.holdouts])
            lines.append(
                f"  leave-one-station-out ({len(self.holdouts)} stations): "
                f"combined score falls {drops.mean():.2f} on average "
                f"(worst {drops.max():.2f}, best {drops.min():.2f}) when the "
                f"station covering a spot is removed"
            )
            for h in sorted(self.holdouts, key=lambda x: -x.drop)[:3]:
                lines.append(
                    f"      station {h.station:2d}: {h.with_station:.2f} -> "
                    f"{h.without_station:.2f} over {h.n_cells} cells"
                )
        return "\n".join(lines)


def _cell_lookup(cells: np.ndarray) -> dict[int, int]:
    return {int(c): i for i, c in enumerate(cells)}


def _yaw_bin(model: MapModel, kf_row: int, yaw_bins: int) -> int:
    yaw = np.degrees(model.heading(model.raw.kf_rot_cw[kf_row, 2])) % 360.0
    return int(yaw / (360.0 / yaw_bins)) % yaw_bins


def at_keyframe_poses(model: MapModel, grid: Grid, result: Result,
                      cfg: Config | None = None) -> PoseCheck:
    """Score the map where it was built. It should agree with itself."""
    min_keypoints = (cfg or DEFAULT).checks.rich_view_keypoints
    lut = _cell_lookup(result.cells)
    t, r, c, m = [], [], [], []
    skipped = 0
    for row in range(model.raw.n_keyframes):
        if model.raw.kf_n_keypts[row] < min_keypoints:
            skipped += 1
            continue
        flat = int(grid.sample_flat(model.ground(model.raw.kf_center[row:row + 1]))[0])
        i = lut.get(flat)
        if i is None:
            skipped += 1
            continue
        y = _yaw_bin(model, row, result.yaw_bins)
        t.append(result.track[i, y])
        r.append(result.reloc[i, y])
        c.append(result.combined[i, y])
        m.append(result.best_matches[i, y])
    return PoseCheck(
        n=len(t), n_skipped=skipped,
        track=np.array(t), reloc=np.array(r),
        combined=np.array(c), matches=np.array(m),
    )


def _cell_centres(grid: Grid, cells: np.ndarray) -> np.ndarray:
    ix, iy = np.divmod(cells, grid.ny)
    return np.column_stack([
        grid.origin[0] + (ix + 0.5) * grid.res,
        grid.origin[1] + (iy + 0.5) * grid.res,
    ])


def decay_with_distance(model: MapModel, grid: Grid, result: Result,
                        edges: tuple[float, ...] = (0, 1, 2, 3, 5, np.inf),
                        ) -> list[DecayBand]:
    """How quality falls off as you walk away from the nearest scan position."""
    xy = _cell_centres(grid, result.cells)
    d, _ = cKDTree(model.ground(model.stations.center)).query(xy)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (d >= lo) & (d < hi)
        if sel.sum() == 0:
            continue
        out.append(DecayBand(
            lo=float(lo), hi=float(hi), n=int(sel.sum()),
            track=float(result.track[sel].mean()),
            reloc=float(result.reloc[sel].mean()),
            combined=float(result.combined[sel].mean()),
        ))
    return out


def leave_one_station_out(model: MapModel, grid: Grid, occ: Occupancy,
                          vis: Visibility, result: Result, height: float,
                          radius: float = 1.5, max_stations: int | None = None,
                          combine: str = "product",
                          cfg: Config | None = None) -> list[Holdout]:
    """Predict each station's own surroundings from a map that never went there.

    The building does not change when a station is dropped, so the floor plan
    and the sightlines are reused; only what the map knows changes.
    """
    xy = _cell_centres(grid, result.cells)
    st_xy = model.ground(model.stations.center)
    order = range(model.stations.n)
    if max_stations is not None and model.stations.n > max_stations:
        order = np.linspace(0, model.stations.n - 1, max_stations).astype(int)

    out: list[Holdout] = []
    for s in order:
        near = np.linalg.norm(xy - st_xy[s], axis=1) <= radius
        if near.sum() < 4:
            continue
        try:
            reduced = loader.without_keyframes(model.raw, model.stations.kf_rows[s])
            sub_model = mapmodel.build(reduced, cfg=cfg, up=model.up)
        except ValueError:
            continue
        sub_vis = vis.subset(near)
        sub = scorers.evaluate(
            sub_model, grid, occ, sub_vis, height=height,
            yaw_bins=result.yaw_bins, combine=combine,
        )
        out.append(Holdout(
            station=int(s), n_cells=int(near.sum()),
            with_station=float(result.combined[near].mean()),
            without_station=float(sub.combined.mean()),
        ))
    return out


SELF_MATCH_YIELD = 0.7
"""BoW yield when the query *is* the map's own keyframe image.

Pinned on the DFL scans: 0.91-1.00 relocalization at their own keyframes; 0.5
reads pessimistic, 1.0 saturates.  Only for this check.  A real camera
against the map matches far worse, and that is ``bow_match_yield``.
"""


def self_match_config(cfg: Config | None) -> Config:
    """Settings for scoring the map against its own keyframe images.

    No domain gap: the query image is the one the map was built from, so the
    BoW yield is the self-match one, every one of the keyframe's landmarks
    projects back onto its own keypoint (the second gate cannot fail), and
    real-run calibration does not apply.  An upper bound, by construction.
    """
    import copy
    c = copy.deepcopy(cfg or DEFAULT)
    c.relocalization.bow_match_yield = SELF_MATCH_YIELD
    c.relocalization.min_valid_obs = 0.0
    c.tracking.calib_a, c.tracking.calib_b = 1.0, 0.0
    return c


def run(model: MapModel, grid: Grid, occ: Occupancy, vis: Visibility,
        result: Result, height: float, holdout: bool = True,
        max_stations: int | None = None, cfg: Config | None = None) -> Report:
    # The own-keyframe check runs in the rendered domain, on just the cells
    # that hold a keyframe, so it costs little.
    kf_cells = np.unique(grid.sample_flat(model.ground(model.raw.kf_center)))
    near = np.isin(vis.cells, kf_cells)
    self_result = scorers.evaluate(
        model, grid, occ, vis.subset(near), height=height,
        yaw_bins=result.yaw_bins, combine=result.combine_mode,
        cfg=self_match_config(cfg),
    ) if near.any() else result
    return Report(
        poses=at_keyframe_poses(model, grid, self_result, cfg),
        decay=decay_with_distance(model, grid, result),
        holdouts=leave_one_station_out(
            model, grid, occ, vis, result, height, max_stations=max_stations,
            combine=result.combine_mode, cfg=cfg,
        ) if holdout else [],
    )
