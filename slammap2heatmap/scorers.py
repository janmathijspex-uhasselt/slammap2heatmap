"""Predict how well a camera would localize, cell by cell and heading by heading.

Three layers, from one pass over the map.

**Tracking** is stella_vslam's own ``landmark::is_observable`` test run against
a virtual camera: can a device that already knows roughly where it is keep a
lock?  On top of the raw count sit two things a count cannot express -- how
*matchable* each visible point is, and whether the visible points are
arranged well enough to pin a pose down at all.

**Relocalization** is the harder question: can a device that knows *nothing*
find itself?  That runs through a different pipeline in stella -- bag-of-words
keyframe retrieval, then descriptor matching, then EPnP inside RANSAC -- and
it fails in different places.  A corridor can be rich in points and still be
hopeless to relocalize in, because both ends look the same.

**Combined** is the product of the two, because that is the real sequence:
you must first find yourself, then stay found.

The decomposition that makes this affordable: of the gates a landmark has to
pass, all but the bearing test are independent of where the camera points.
So candidates are filtered once per cell, and every heading is recovered from
one histogram over bearing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import poisson

from .config import DEFAULT, Config
from .grid import Grid
from .mapmodel import MapModel
from .occupancy import Occupancy
from .visibility import Visibility



@dataclass
class Result:
    """Every layer, for every swept cell and heading."""

    grid: Grid
    cells: np.ndarray          # (C,) flat grid indices
    yaw_bins: int
    height: float

    n_eff: np.ndarray          # (C, yaw) float32, weighted trackable points
    n_raw: np.ndarray          # (C, yaw) float32, unweighted count
    sigma_pos: np.ndarray      # (C, yaw) float32, metres: geometry + systematic floor
    sigma_geom: np.ndarray     # (C, yaw) float32, metres: geometry alone
    expected_matched: np.ndarray  # (C, yaw) float32, n_eff in real matched landmarks
    track_raw: np.ndarray      # (C, yaw) float32, 0..1, before calibration
    track: np.ndarray          # (C, yaw) float32, 0..1
    reloc: np.ndarray          # (C, yaw) float32, 0..1, per attempt (frame)
    reloc_within: np.ndarray   # (C, yaw) float32, 0..1, within find_within_s
    reloc_time_s: np.ndarray   # (C, yaw) float32, expected seconds to relocalize
    best_matches: np.ndarray   # (C, yaw) float32, expected matches, best keyframe
    combined: np.ndarray       # (C, yaw) float32, 0..1
    combine_mode: str

    def yaw_degrees(self) -> np.ndarray:
        return (np.arange(self.yaw_bins) + 0.5) * 360.0 / self.yaw_bins

    def summary(self) -> str:
        return (
            f"{len(self.cells)} cells x {self.yaw_bins} headings: "
            f"trackable points median {np.median(self.n_eff):.0f}, "
            f"track {self.track.mean():.2f}, reloc {self.reloc.mean():.2f}, "
            f"combined {self.combined.mean():.2f}; "
            f"usable facing anywhere {100 * (self.combined.min(axis=1) > 0.5).mean():.0f}%"
        )


def calibrate_track(p: np.ndarray, a: float, b: float) -> np.ndarray:
    """Platt-style recalibration: sigmoid(a * logit(p) + b).

    Two parameters, monotone, and the identity at a=1, b=0. Chosen over an
    isotonic fit because real runs give few losses to learn from, and a step
    function fitted to a dozen events would mostly be noise.
    """
    if a == 1.0 and b == 0.0:
        return p.astype(np.float32)
    q = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    z = a * np.log(q / (1 - q)) + b
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _box_filter_circular(hist: np.ndarray, width: int) -> np.ndarray:
    """Sum each circular window of ``width`` bins, for every row at once."""
    n = hist.shape[-1]
    half = width // 2
    padded = np.concatenate([hist[..., -half:], hist, hist[..., :width]], axis=-1)
    csum = np.cumsum(padded, axis=-1, dtype=np.float64)
    csum = np.concatenate([np.zeros(csum.shape[:-1] + (1,)), csum], axis=-1)
    return (csum[..., width:width + n] - csum[..., :n]).astype(np.float32)


def _observation_index(model: MapModel, usable: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compact landmark -> (observing keyframes, their viewing distances).

    Returned as a CSR-style triple so the per-cell inner loop can expand a
    list of landmarks into a list of keyframes without any Python looping.
    """
    raw = model.raw
    remap = np.full(len(usable), -1, dtype=np.int64)
    remap[usable] = np.arange(int(usable.sum()))

    lm = remap[raw.obs_lm]
    keep = lm >= 0
    lm, kf = lm[keep], raw.obs_kf[keep]
    dist = np.linalg.norm(
        raw.lm_pos_w[raw.obs_lm[keep]] - raw.kf_center[kf], axis=1
    ).astype(np.float32)

    order = np.argsort(lm, kind="stable")
    lm, kf, dist = lm[order], kf[order], dist[order]
    starts = np.searchsorted(lm, np.arange(int(usable.sum()) + 1))
    return starts.astype(np.int64), kf, dist


def _expand(starts: np.ndarray, li: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ragged gather: landmark rows -> observation rows, plus which input row."""
    counts = starts[li + 1] - starts[li]
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    src = np.repeat(np.arange(len(li)), counts)
    offset = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    return starts[li][src] + offset, src


def _circular_window(start: np.ndarray, width: int, n: int,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deltas that, once accumulated and cumulatively summed, fill a window.

    Each landmark is seen from a contiguous arc of headings.  Writing one +1
    and one -1 instead of one entry per heading keeps the scatter small, and
    a cumulative sum along the heading axis expands them again.
    """
    end = (start + width) % n
    wraps = end <= start
    return start % n, end, wraps


def evaluate(model: MapModel, grid: Grid, occ: Occupancy, vis: Visibility,
             height: float, yaw_bins: int = 16,
             use_occlusion: bool = True,
             combine: str = "product",
             cfg: Config | None = None,
             chunk: int = 256) -> Result:
    """Score every swept cell for tracking, relocalization, and both together."""
    conf = cfg or DEFAULT
    t_cfg, r_cfg = conf.tracking, conf.relocalization
    pixel_noise = t_cfg.pixel_noise_px
    cal = conf.calibration
    bearing_bins = t_cfg.bearing_bins
    cam = model.camera
    cells = vis.cells
    raw = model.raw

    usable = model.weight > 0
    lm_xy = model.ground(raw.lm_pos_w)[usable]
    lm_h = model.height(raw.lm_pos_w)[usable]
    w = model.weight[usable].astype(np.float64)
    normal_xy = model.ground(model.normal)[usable]
    normal_h = model.height(model.normal)[usable]
    d_lo = model.d_min[usable] * t_cfg.scale_margin_lo
    d_hi = model.d_max[usable] * t_cfg.scale_margin_hi
    d_ref = model.d_ref[usable]

    octave_scale = np.where(d_ref > 0, model.d_max[usable] / np.maximum(d_ref, 1e-6), 1.0)
    sigma_px = pixel_noise * octave_scale
    # The map's own point error, measured from its duplicate landmarks (or
    # set in the config). Constant with range: the DFL scans show only weak
    # growth with distance, nothing like the d^2 of a depth camera.
    sigma_lm = np.full(len(d_ref), model.landmark_sigma, dtype=np.float64)

    obs_starts, obs_kf, obs_dist = _observation_index(model, usable)
    K = raw.n_keyframes

    ix, iy = np.divmod(cells, grid.ny)
    cx = grid.origin[0] + (ix + 0.5) * grid.res
    cy = grid.origin[1] + (iy + 0.5) * grid.res

    tan_v = np.tan(np.radians(cam.vfov_deg / 2.0))
    fov_bins = max(1, int(round(cam.hfov_deg / 360.0 * bearing_bins)))
    focal = cam.focal_px

    # How far a keyframe's heading is from each of ours.  Independent of
    # position, so it is a small table rather than part of the inner loop.
    kf_dir = model.ground(raw.kf_rot_cw[:, 2, :])
    kf_dir /= np.maximum(np.linalg.norm(kf_dir, axis=1, keepdims=True), 1e-9)
    yaw_rad = np.radians((np.arange(yaw_bins) + 0.5) * 360.0 / yaw_bins)
    q_dir = np.stack([np.cos(yaw_rad), np.sin(yaw_rad)], axis=1)
    cos_full = np.cos(np.radians(r_cfg.viewpoint_full_deg))
    cos_dead = np.cos(np.radians(r_cfg.viewpoint_dead_deg))
    a_angle = np.clip(
        (q_dir @ kf_dir.T - cos_dead) / (cos_full - cos_dead), 0.0, 1.0
    )                                                     # (yaw, K)

    station_kfs = model.stations.kf_rows

    tree = cKDTree(lm_xy)
    max_r = float(min(vis.max_range, d_hi.max() if len(d_hi) else 0.0))

    C = len(cells)
    n_eff = np.zeros((C, yaw_bins), dtype=np.float32)
    n_raw = np.zeros((C, yaw_bins), dtype=np.float32)
    sigma_pos = np.full((C, yaw_bins), t_cfg.sigma_pos_max_m, dtype=np.float32)
    reloc = np.zeros((C, yaw_bins), dtype=np.float32)
    best_matches = np.zeros((C, yaw_bins), dtype=np.float32)

    tri = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    yaw_width = int(round(cam.hfov_deg / (360.0 / yaw_bins)))
    yaw_width = max(1, min(yaw_width, yaw_bins))

    for lo in range(0, C, chunk):
        hi = min(lo + chunk, C)
        nc = hi - lo
        idx_lists = tree.query_ball_point(
            np.column_stack([cx[lo:hi], cy[lo:hi]]), max_r, return_sorted=False
        )
        counts = np.fromiter((len(v) for v in idx_lists), dtype=np.int64, count=nc)
        if counts.sum() == 0:
            continue
        li = np.concatenate([np.asarray(v, dtype=np.int64) for v in idx_lists if v])
        ci = np.repeat(np.arange(nc), counts)

        dx = lm_xy[li, 0] - cx[lo:hi][ci]
        dy = lm_xy[li, 1] - cy[lo:hi][ci]
        dz = lm_h[li] - height
        rho = np.hypot(dx, dy)
        d3 = np.maximum(np.sqrt(rho * rho + dz * dz), 1e-6)

        keep = (d3 >= d_lo[li]) & (d3 <= d_hi[li])            # ORB scale band
        keep &= rho > 1e-3
        keep &= np.abs(dz) <= tan_v * np.maximum(rho, 1e-6)   # vertical FOV
        cos_view = -(dx * normal_xy[li, 0] + dy * normal_xy[li, 1]
                     + dz * normal_h[li]) / d3
        keep &= cos_view >= t_cfg.view_cone_cos                     # facing us at all

        bearing = np.arctan2(dy, dx)
        if use_occlusion:
            bb = vis.bearing_bin(bearing)
            keep &= rho <= vis.free_range[lo:hi][ci, bb] + grid.res

        if not keep.any():
            continue
        ci, li = ci[keep], li[keep]
        d3, bearing = d3[keep], bearing[keep]
        dxk, dyk, dzk = dx[keep], dy[keep], dz[keep]

        # ---- tracking: bearing histogram, then one FOV convolution --------
        b = np.column_stack([dxk / d3, dyk / d3, dzk / d3])
        sig2 = sigma_px[li] ** 2 + (focal * sigma_lm[li] / d3) ** 2
        k_info = w[li] * focal ** 2 / (sig2 * d3 ** 2)

        bin_i = np.minimum(
            ((bearing / (2 * np.pi)) % 1.0 * bearing_bins).astype(np.int64),
            bearing_bins - 1,
        )
        flat = ci * bearing_bins + bin_i
        size = nc * bearing_bins

        acc = np.empty((8, nc, bearing_bins), dtype=np.float64)
        acc[0] = np.bincount(flat, weights=w[li], minlength=size).reshape(nc, -1)
        acc[1] = np.bincount(flat, minlength=size).reshape(nc, -1)
        for t, (p, q) in enumerate(tri):
            val = k_info * ((1.0 if p == q else 0.0) - b[:, p] * b[:, q])
            acc[2 + t] = np.bincount(flat, weights=val, minlength=size).reshape(nc, -1)

        conv = _box_filter_circular(acc, fov_bins)
        take = ((np.arange(yaw_bins) + 0.5) * bearing_bins / yaw_bins).astype(int)
        conv = conv[:, :, take]

        n_eff[lo:hi] = conv[0]
        n_raw[lo:hi] = conv[1]

        lam = np.zeros((nc, yaw_bins, 3, 3), dtype=np.float64)
        for t, (p, q) in enumerate(tri):
            lam[..., p, q] = conv[2 + t]
            lam[..., q, p] = conv[2 + t]
        for a in range(3):
            lam[..., a, a] += 1e-9
        var = np.einsum("...ii->...", np.linalg.inv(lam))
        sigma_pos[lo:hi] = np.sqrt(np.clip(var, 0.0, t_cfg.sigma_pos_max_m ** 2))

        # ---- relocalization: overlap with each keyframe, per heading ------
        obs_rows, src = _expand(obs_starts, li)
        if obs_rows.size == 0:
            continue
        o_cell = ci[src]
        o_kf = obs_kf[obs_rows]
        o_dk = obs_dist[obs_rows]
        o_dp = d3[src]

        # The arc of headings that sees this landmark, as two deltas.
        start_bin = np.ceil(
            (np.degrees(bearing[src]) % 360.0 - cam.hfov_deg / 2.0)
            / (360.0 / yaw_bins) - 0.5
        ).astype(np.int64)
        s, e, wraps = _circular_window(start_bin, yaw_width, yaw_bins)

        base = (o_cell * K + o_kf) * yaw_bins
        size_r = nc * K * yaw_bins

        def scatter(values: np.ndarray) -> np.ndarray:
            d = np.bincount(base + s, weights=values, minlength=size_r)
            d -= np.bincount(base + e, weights=values, minlength=size_r)
            d += np.bincount(base[wraps], weights=values[wraps], minlength=size_r)
            return np.cumsum(d.reshape(nc, K, yaw_bins), axis=2)

        # Weighted, not counted: an ambiguous descriptor or one of several
        # duplicates of the same corner will not win a match, and the same
        # weighting is what the tracking layer counts.
        m = scatter(w[li][src])                 # shared, matchable landmarks
        sum_dp = scatter(o_dp.astype(np.float64))
        sum_dk = scatter(o_dk.astype(np.float64))

        # Same structure, but seen from how far away compared with the
        # keyframe that stored it?  Past a ratio of ~1.5 the descriptors
        # sit at different pyramid levels and stop matching.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(sum_dk > 0, sum_dp / np.maximum(sum_dk, 1e-9), 1.0)
        a_scale = np.exp(
            -np.log(np.clip(ratio, 1e-3, 1e3)) ** 2
            / (2.0 * np.log(r_cfg.scale_tolerance) ** 2)
        )

        # Gate 1, the hard one: bag-of-words matches against one keyframe.
        # This is where real camera images and a rendered or scan-built map
        # part ways, which is what bow_match_yield absorbs.
        # The cumulative sum that builds m leaves -1e-17 where it means 0, and
        # a Poisson mean must not be negative.
        m = np.maximum(m, 0.0)
        matches = m * a_scale * a_angle.T[None, :, :] * r_cfg.bow_match_yield
        # Chance of at least min_bow_matches when matches are expected: a
        # Poisson tail. The logistic this replaces kept ~5 % success with
        # nothing to match at all, which put a floor under the whole layer
        # that real runs (under 1 % per attempt) could not get below.
        p_bow = poisson.sf(np.ceil(r_cfg.min_bow_matches) - 1, matches)
        # Gate 2: after the PnP pose, stella projects the candidate's
        # landmarks and needs min_valid_obs to survive refinement. That is
        # guided search, like tracking, so it uses the tracking yield.
        projected = m * a_scale * cal.matched_per_neff
        p_valid = poisson.sf(np.ceil(r_cfg.min_valid_obs) - 1, projected)
        p_kf = p_bow * p_valid

        # Keyframes of one station fail together, and so -- the first
        # real-run check showed -- do separate stations: same frame, same
        # blur, same exposure. So by default the best chance is the chance.
        p_station = np.stack(
            [p_kf[:, rows, :].max(axis=1) for rows in station_kfs], axis=1
        )
        if r_cfg.pooling == "max":
            reloc[lo:hi] = p_station.max(axis=1)
        else:
            top = np.sort(p_station, axis=1)[:, -r_cfg.candidates:, :]
            reloc[lo:hi] = 1.0 - np.prod(1.0 - top, axis=1)
        best_matches[lo:hi] = matches.max(axis=1)

    # Geometry alone, scaled for real-image keypoint noise ...
    sigma_geom = np.clip(sigma_pos * t_cfg.sigma_scale, 0.0, t_cfg.sigma_pos_max_m)
    # ... plus what no number of landmarks averages away.
    sigma_pos = np.sqrt(sigma_geom ** 2 + t_cfg.sigma_floor_m ** 2)
    np.clip(sigma_pos, 0.0, t_cfg.sigma_pos_max_m, out=sigma_pos)

    with np.errstate(divide="ignore"):
        logit = t_cfg.track_sharpness * (
            np.log(np.maximum(n_eff, 1e-3)) - np.log(t_cfg.n_tracked_marginal)
        )
    p_enough = 1.0 / (1.0 + np.exp(-logit))
    # Degenerate geometry, not the systematic floor: an offset map is still
    # perfectly trackable.
    conditioning = 1.0 / (1.0 + (sigma_geom / t_cfg.sigma_pos_good_m) ** 2)
    track_raw = (p_enough * conditioning).astype(np.float32)
    track = calibrate_track(track_raw, t_cfg.calib_a, t_cfg.calib_b)

    # stella attempts relocalization on every frame while lost. A chance per
    # attempt is what real runs calibrate; a person standing there cares how
    # long it takes, and whether it happens within a reasonable wait.
    fps = cam.fps if cam.fps > 0 else 30.0
    attempts = max(1.0, r_cfg.find_within_s * fps)
    p_attempt = np.clip(reloc, 0.0, 1.0)
    reloc_within = 1.0 - (1.0 - p_attempt) ** attempts
    reloc_time_s = np.clip(1.0 / (np.maximum(p_attempt, 1e-12) * fps),
                           1.0 / fps, 3600.0)

    if combine == "product":
        combined = track * reloc_within
    elif combine == "min":
        combined = np.minimum(track, reloc_within)
    elif combine == "sum":
        combined = 0.5 * (track + reloc_within)
    else:
        raise ValueError(f"unknown combine mode {combine!r}")

    return Result(
        grid=grid, cells=cells, yaw_bins=yaw_bins, height=height,
        n_eff=n_eff, n_raw=n_raw,
        sigma_pos=sigma_pos.astype(np.float32),
        sigma_geom=sigma_geom.astype(np.float32),
        expected_matched=(n_eff * cal.matched_per_neff).astype(np.float32),
        track_raw=track_raw, track=track, reloc=reloc.astype(np.float32),
        reloc_within=reloc_within.astype(np.float32),
        reloc_time_s=reloc_time_s.astype(np.float32),
        best_matches=best_matches, combined=combined.astype(np.float32),
        combine_mode=combine,
    )
