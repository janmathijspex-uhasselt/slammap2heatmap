"""Recover a floor plan from the map, and with it the walls that block sight.

Nothing in a stella map says where the walls are.  But every landmark was
seen from a known camera, and that sighting is evidence about two things at
once: the space along the ray was empty, and something solid sits at the far
end.  Accumulating that over every observation reconstructs the floor plan.

It is worth the effort twice over: it is the backdrop that makes a heatmap
readable, and it is the occlusion mask that stops scores leaking through
walls into rooms nobody scanned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT, Config
from .grid import Grid
from .mapmodel import MapModel

UNKNOWN, FREE, OCCUPIED = 0, 1, 2

# Every threshold that shapes this lives in config.py, so that changing one
# has an effect rather than editing a copy the code no longer reads.

def _synthetic_scans(model: MapModel, c) -> tuple[np.ndarray, np.ndarray]:
    """Collapse each keyframe's observations into one range per bearing.

    A stella map has no range sensor, but a keyframe's landmarks are a sparse
    depth image, and at eye level that is very nearly a laser scan.  Reducing
    each bearing to a single range before carving is what makes the result a
    floor plan instead of a starburst: a ray then represents a direction that
    was actually measured, not one landmark that happened to be found.

    Returns beam origins and endpoints in ground coordinates.
    """
    raw = model.raw
    h_lm = model.height(raw.lm_pos_w)[raw.obs_lm]
    h_cam = model.height(raw.kf_center)[raw.obs_kf]
    keep = (
        (h_lm > h_cam - c.wall_band_below_m)
        & (h_lm < h_cam + c.wall_band_above_m)
        & ~model.outlier[raw.obs_lm]
    )
    if not keep.any():
        raise ValueError(
            "no observations fall in the wall band; the map may be empty or "
            "the up axis wrong"
        )

    cam = model.ground(raw.kf_center)[raw.obs_kf[keep]]
    lm = model.ground(raw.lm_pos_w)[raw.obs_lm[keep]]
    delta = lm - cam
    rng = np.hypot(delta[:, 0], delta[:, 1])
    bearing = np.degrees(np.arctan2(delta[:, 1], delta[:, 0])) % 360.0

    n_bins = int(round(360.0 / c.scan_bin_deg))
    beam = (bearing / 360.0 * n_bins).astype(np.int64) % n_bins
    key = raw.obs_kf[keep] * n_bins + beam

    order = np.lexsort((rng, key))
    key_s, rng_s = key[order], rng[order]
    starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
    counts = np.diff(np.r_[starts, len(key_s)])

    good = counts >= c.scan_min_hits
    starts, counts = starts[good], counts[good]
    # Ranges are already sorted within each group, so the percentile is a pick.
    pick = starts + np.minimum(
        (counts * c.scan_percentile / 100.0).astype(np.int64), counts - 1
    )
    ranges = rng_s[pick]

    kf_row = key_s[starts] // n_bins
    beam_id = key_s[starts] % n_bins
    theta = np.radians((beam_id + 0.5) * c.scan_bin_deg)

    origin = model.ground(raw.kf_center)[kf_row]
    end = origin + np.column_stack([np.cos(theta), np.sin(theta)]) * ranges[:, None]
    return origin, end


@dataclass
class Occupancy:
    grid: Grid
    log_odds: np.ndarray      # (nx, ny) float32
    state: np.ndarray         # (nx, ny) uint8, one of UNKNOWN / FREE / OCCUPIED
    passes: np.ndarray        # (nx, ny) int32, distinct sightlines crossing
    n_rays: int

    @property
    def free(self) -> np.ndarray:
        return self.state == FREE

    @property
    def occupied(self) -> np.ndarray:
        return self.state == OCCUPIED

    def summary(self) -> str:
        total = self.state.size
        return (
            f"{self.n_rays} scan beams -> "
            f"{100 * self.free.mean():.0f}% free, "
            f"{100 * self.occupied.mean():.0f}% wall, "
            f"{100 * (self.state == UNKNOWN).mean():.0f}% unseen "
            f"of {total} cells at {self.grid.res:.2f} m"
        )


def build(model: MapModel, grid: Grid, cfg: Config | None = None,
          chunk: int = 8192) -> Occupancy:
    """Carve free space along every sightline and mark where it stops."""
    c = (cfg or DEFAULT).occupancy
    start, end = _synthetic_scans(model, c)

    log_odds = np.zeros(grid.size, dtype=np.float32)
    passes = np.zeros(grid.size, dtype=np.int32)
    step = grid.res * 0.5

    for i in range(0, len(start), chunk):
        a, b = start[i:i + chunk], end[i:i + chunk]
        d = np.linalg.norm(b - a, axis=1)
        unit = np.divide(b - a, np.maximum(d, 1e-9)[:, None])

        # The endpoint is structure wherever it is; only the carving is capped.
        hit = grid.sample_flat(b)
        np.add.at(log_odds, hit[hit >= 0], c.log_odds_occupied)

        # Stop half a cell short so the endpoint's own cell is not carved free.
        reach = np.clip(d - grid.res, 0.0, c.max_ray_m)
        n_steps = int(np.ceil(reach.max() / step)) + 1
        t = np.arange(n_steps, dtype=np.float32) * step        # (S,)
        valid = t[None, :] <= reach[:, None]                   # (R, S)
        pts = a[:, None, :] + unit[:, None, :] * t[None, :, None]

        flat = grid.sample_flat(pts.reshape(-1, 2)).reshape(valid.shape)
        ok = valid & (flat >= 0)
        # One ray must not count twice for the same cell: it is one sample.
        cells = np.where(ok, flat, -1)
        cells.sort(axis=1)
        first = np.empty_like(cells, dtype=bool)
        first[:, 0] = True
        np.not_equal(cells[:, 1:], cells[:, :-1], out=first[:, 1:])
        sel = cells[first & (cells >= 0)]
        np.add.at(log_odds, sel, c.log_odds_free)
        np.add.at(passes, sel, 1)

    np.clip(log_odds, -c.log_odds_clamp, c.log_odds_clamp, out=log_odds)
    log_odds = log_odds.reshape(grid.nx, grid.ny)
    passes = passes.reshape(grid.nx, grid.ny)

    state = np.full(log_odds.shape, UNKNOWN, dtype=np.uint8)
    state[(log_odds < 0) & (passes >= c.min_free_passes)] = FREE
    state[log_odds >= c.occupied_threshold] = OCCUPIED

    return Occupancy(grid=grid, log_odds=log_odds, state=state, passes=passes,
                     n_rays=len(start))
