"""Prove that a map was decoded correctly, before anything is inferred from it.

The whole tool rests on one assumption: that ``rot_cw`` is a quaternion in
(x, y, z, w) order and ``trans_cw`` its matching world -> camera translation.
If a future map is written by a different version of stella, or by a
different exporter, that assumption can silently break -- and every heatmap
after it would be confidently wrong.

So we check it against the file's own data.  Reprojecting each landmark
through the pose of a keyframe that observed it must land on the stored
keypoint, and the resulting camera-frame z must equal the stored RGBD depth.
Both are redundant information in the file, which is exactly what makes them
usable as a test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .loader import RawMap
from .mapmodel import MapModel

from .config import DEFAULT, Config


@dataclass
class GeometryCheck:
    tolerance_px: float
    tolerance_depth_rel: float
    tolerance_behind: float
    n: int
    n_total: int
    behind: int
    reproj_med: float
    reproj_p95: float
    reproj_max: float
    depth_med: float              # metres, NaN when the map holds no depth
    depth_rel: float              # median |error| / depth, NaN without depth

    @property
    def behind_frac(self) -> float:
        return self.behind / self.n if self.n else 1.0

    @property
    def ok(self) -> bool:
        """Whether the pose convention holds.

        Judged on medians, because the question is "was this file decoded
        correctly", not "is every landmark in it good".
        """
        return (
            self.n > 0
            and self.behind_frac <= self.tolerance_behind
            and self.reproj_med <= self.tolerance_px
            and (not np.isfinite(self.depth_rel)
                 or self.depth_rel <= self.tolerance_depth_rel)
        )

    def describe(self) -> str:
        verdict = "OK" if self.ok else "FAILED"
        scope = (
            f"{self.n} observations"
            if self.n == self.n_total
            else f"{self.n} of {self.n_total} observations sampled"
        )
        if np.isfinite(self.depth_rel):
            depth = (f"depth median {self.depth_med * 1000:.1f} mm "
                     f"({self.depth_rel * 100:.2f}% of range)")
        else:
            depth = "no depth stored (monocular)"
        return (
            f"geometry {verdict}: {scope}, "
            f"reprojection median {self.reproj_med:.2f} px "
            f"(p95 {self.reproj_p95:.2f}), {depth}, "
            f"{self.behind} behind camera"
        )


def check_geometry(raw: RawMap, sample: int | None = 50_000,
                   seed: int = 0, cfg: Config | None = None) -> GeometryCheck:
    """Reproject observations through their own keyframe poses."""
    c = (cfg or DEFAULT).checks
    tol = dict(tolerance_px=c.reproj_tolerance_px,
               tolerance_depth_rel=c.depth_tolerance_rel,
               tolerance_behind=c.behind_tolerance)
    empty = dict(reproj_med=np.inf, reproj_p95=np.inf, reproj_max=np.inf,
                 depth_med=np.nan, depth_rel=np.nan)
    n_total = raw.n_observations
    if n_total == 0:
        return GeometryCheck(**tol, n=0, n_total=0, behind=0, **empty)

    sel = np.arange(n_total)
    if sample is not None and n_total > sample:
        sel = np.random.default_rng(seed).choice(n_total, sample, replace=False)

    kf, lm = raw.obs_kf[sel], raw.obs_lm[sel]
    pc = np.einsum("nij,nj->ni", raw.kf_rot_cw[kf], raw.lm_pos_w[lm]) + raw.kf_trans_cw[kf]

    cam = raw.camera
    uv, front = cam.project(pc)
    behind = int((~front).sum())
    if not front.any():
        return GeometryCheck(**tol, n=len(sel), n_total=n_total, behind=behind,
                             **empty)

    err = np.linalg.norm(uv[front] - raw.obs_pt[sel][front], axis=1)
    if cam.is_spherical:
        # Longitude wraps at the image seam.
        du = np.abs(uv[front, 0] - raw.obs_pt[sel][front, 0])
        du = np.minimum(du, cam.cols - du)
        err = np.hypot(du, uv[front, 1] - raw.obs_pt[sel][front, 1])

    # Stored depth is camera-frame z for pinhole cameras.
    z = pc[front, 2]
    d = raw.obs_depth[sel][front]
    ok = np.isfinite(d) & (z > 0)
    if ok.any():
        derr = np.abs(z[ok] - d[ok])
        depth_med = float(np.median(derr))
        depth_rel = float(np.median(derr / z[ok]))
    else:
        depth_med = depth_rel = np.nan

    return GeometryCheck(
        **tol,
        n=len(sel),
        n_total=n_total,
        behind=behind,
        reproj_med=float(np.median(err)),
        reproj_p95=float(np.percentile(err, 95)),
        reproj_max=float(err.max()),
        depth_med=depth_med,
        depth_rel=depth_rel,
    )


def summarise(model: MapModel, check: GeometryCheck) -> str:
    """A human-readable report on one map."""
    raw = model.raw
    cam = model.camera
    st = model.stations
    spacing = st.spacing()

    g = model.ground(raw.lm_pos_w)
    lo, hi = np.percentile(g, [1, 99], axis=0)
    lm_h = model.height(raw.lm_pos_w)
    kf_per_station = np.array([len(r) for r in st.kf_rows])
    n_distinct = len(np.unique(model.dup_group))
    n_loops = sum(len(e) for e in raw.kf_loop_edges)

    lines = [
        f"{raw.path.name}",
        f"  {check.describe()}",
        f"  camera      {cam.model_type}/{cam.setup_type} {cam.cols}x{cam.rows} "
        f"fov {cam.hfov_deg:.1f}deg x {cam.vfov_deg:.1f}deg  "
        f"orb {model.orb.num_levels} levels x{model.orb.scale_factor:.2f}",
        f"  keyframes   {raw.n_keyframes} in {st.n} stations "
        f"({kf_per_station.min()}-{kf_per_station.max()} each), "
        f"spacing median {np.median(spacing):.2f} m" if st.n > 1 else
        f"  keyframes   {raw.n_keyframes} in {st.n} station",
        f"  landmarks   {raw.n_landmarks} stored, {n_distinct} distinct "
        f"({100 * (1 - n_distinct / raw.n_landmarks):.0f}% duplicates), "
        f"{raw.n_observations} observations",
        f"  frame       up={model.up_axis}  floor {model.floor_h:.2f} m  "
        f"ceiling {model.ceiling_h:.2f} m  camera height {model.station_h:.2f} m",
        f"  extent      {hi[0] - lo[0]:.1f} x {hi[1] - lo[1]:.1f} m "
        f"(1-99 pct), height span {np.percentile(lm_h, 98) - np.percentile(lm_h, 2):.1f} m",
        f"  defects     {int(model.outlier.sum())} landmarks unusable "
        f"({100 * model.outlier.mean():.2f}%: behind a camera that observed them, "
        f"or reprojecting badly against their own keyframes)",
        f"  quality     weight median {np.median(model.weight):.3f}, "
        f"ambiguous {100 * (model.ambiguity > 0.5).mean():.0f}%, "
        f"observed from >1 station {100 * (model.n_stations > 1).mean():.0f}%",
    ]

    warn = []
    if not raw.is_metric:
        warn.append("MONOCULAR, NO SCALE - lengths are in arbitrary map units, so "
                    "every distance-based setting is meaningless. Measure one "
                    "known distance and pass --scale (true metres / map units)")
    elif raw.scale_given and raw.scale != 1.0:
        warn.append(f"lengths rescaled by {raw.scale:g} at load")
    if not np.allclose(np.abs(model.up), np.round(np.abs(model.up))):
        tilt = np.degrees(np.arccos(np.max(np.abs(model.up))))
        warn.append(f"world frame is tilted {tilt:.1f} deg from gravity "
                    f"(normal for a hand-held run); aligned to the measured up")
    if cam.model_type == "Fisheye":
        warn.append("fisheye field of view approximated as equidistant; "
                    "lens distortion beyond that is ignored")
    if model.counters_informative:
        warn.append("n_vis/n_fnd counters are populated and used to weight "
                    "landmarks by how reliably the tracker found them")
    if (raw.lm_n_vis <= 1).all() and (raw.lm_n_fnd <= 1).all():
        warn.append("n_vis/n_fnd are 1 for every landmark - stored quality "
                    "counters carry no information, rebuilding from lm_ids")
    if n_loops == 0:
        warn.append("no loop edges - global consistency cannot be assessed; "
                    "results describe local localizability only")
    if raw.n_dangling:
        warn.append(f"{raw.n_dangling} keypoints referenced absent landmarks (dropped)")
    ts = np.sort(raw.kf_ts)
    if len(ts) > 2 and np.allclose(np.diff(ts), np.diff(ts)[0], atol=1e-6):
        warn.append("keyframe timestamps are perfectly uniform - synthetic, "
                    "not capture time; temporal reasoning disabled")
    for w in warn:
        lines.append(f"  note        {w}")

    return "\n".join(lines)
