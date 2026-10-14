"""Check the heatmap against real localization runs, and calibrate it on them.

``validate`` can only score the map against its own keyframe images, where
success is guaranteed and there is no gap between a rendered image and a real
camera frame.  That pins the *shape* of the model, not its numbers.  This
module takes what actually happened -- every frame of a real run, at its
ground-truth pose, with stella's own tracked/lost state, landmark count and
position error -- and asks each layer the question it claims to answer:

* **landmarks**: does ``n_eff`` predict how many landmarks stella matched?
* **tracking**: of the frames that were *already tracking*, which kept the
  lock for ``horizon_s``?  Not "which frames had a pose": in the rooms with
  ground truth most lost frames are startup periods where stella had not
  relocalized yet, and counting those against tracking blames the wrong
  layer.
* **relocalization**: of the frames where stella was lost, which one brought
  it back?  stella tries every frame, so each lost frame is one failed
  attempt.
* **position**: is the predicted sigma the size of the real error?

Then it fits the parameters that real images decide and the map cannot:
``matched_per_neff``, the tracking calibration, ``bow_match_yield``, and the
position-error floor and scale.  The fitted values are written to a config
file, so the next heatmap of a similar map starts from real-world numbers.

The frames file is the one the Qualisys alignment app writes (one row per
video frame); see ``read_frames`` for the columns used.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

from . import scorers
from .config import Config
from .grid import Grid
from .mapmodel import MapModel
from .occupancy import Occupancy
from .scorers import Result
from .visibility import Visibility

BLOCK_S = 2.0
"""Bootstrap block length. Neighbouring frames are not independent."""

N_BOOT = 400

YIELD_GRID = np.geomspace(0.001, 1.0, 61)
"""Candidate BoW yields for the relocalization fit."""

PLATT_PRIOR = 2.0
"""Pull of the tracking calibration towards the identity.

Real runs give a dozen or so losses. Without a prior, a two-parameter fit to
that is free to do something extreme; with it, the data has to argue for
moving away from the uncalibrated model.
"""

MAXWELL_95 = 2.7955 / np.sqrt(3.0)
"""95 % quantile of a 3D error, in units of its RMS sigma (about 1.61)."""


# --------------------------------------------------------------------------
# input

@dataclass
class Frames:
    """One row per video frame, from one or more runs."""

    run: np.ndarray            # (N,) str
    t: np.ndarray              # (N,) float, seconds
    source: np.ndarray         # (N,) str: ground_truth / slam / none
    xy: np.ndarray             # (N, 2) map ground coordinates
    heading_deg: np.ndarray    # (N,) optical axis heading
    tracked: np.ndarray        # (N,) bool
    n_matched: np.ndarray      # (N,) float
    error_m: np.ndarray        # (N,) float, NaN without ground truth
    their_cell: np.ndarray     # (N,) int, -1 if absent
    their_bin: np.ndarray      # (N,) int, -1 if absent
    paths: list[str]

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def gt(self) -> np.ndarray:
        return self.source == "ground_truth"

    @property
    def posed(self) -> np.ndarray:
        return np.isin(self.source, ("ground_truth", "slam"))


def _num(v: str, default=np.nan) -> float:
    try:
        return float(v) if v not in ("", None) else default
    except ValueError:
        return default


def read_frames(paths: list[str | Path]) -> Frames:
    """Read one or more frames.csv files.

    Columns used: ``run``, ``t``, ``pose_source``, ``x``, ``y``,
    ``heading_deg``, ``tracked``, ``n_matched``, ``position_error_m``, and
    optionally ``cell_row`` / ``heading_bin`` for a convention cross-check.
    Rows are kept in file order within each run, which must be time order.
    """
    cols: dict[str, list] = {k: [] for k in (
        "run", "t", "source", "x", "y", "h", "tr", "nm", "err", "cell", "bin")}
    for p in paths:
        with open(p, newline="") as fh:
            reader = csv.DictReader(fh)
            need = {"run", "t", "pose_source", "x", "y", "heading_deg", "tracked"}
            missing = need - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{p}: missing columns {', '.join(sorted(missing))}")
            for r in reader:
                cols["run"].append(r["run"])
                cols["t"].append(_num(r["t"]))
                cols["source"].append(r["pose_source"])
                cols["x"].append(_num(r["x"]))
                cols["y"].append(_num(r["y"]))
                cols["h"].append(_num(r["heading_deg"]))
                cols["tr"].append(r["tracked"] in ("1", "true", "True"))
                cols["nm"].append(_num(r.get("n_matched", ""), 0.0))
                cols["err"].append(_num(r.get("position_error_m", "")))
                cols["cell"].append(int(_num(r.get("cell_row", ""), -1)))
                cols["bin"].append(int(_num(r.get("heading_bin", ""), -1)))
    return Frames(
        run=np.array(cols["run"]), t=np.array(cols["t"]),
        source=np.array(cols["source"]),
        xy=np.column_stack([cols["x"], cols["y"]]),
        heading_deg=np.array(cols["h"]), tracked=np.array(cols["tr"]),
        n_matched=np.array(cols["nm"], dtype=float),
        error_m=np.array(cols["err"], dtype=float),
        their_cell=np.array(cols["cell"]), their_bin=np.array(cols["bin"]),
        paths=[str(p) for p in paths],
    )


# --------------------------------------------------------------------------
# statistics

def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Chance a random positive outscores a random negative (ties count half)."""
    label = label.astype(bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks over ties
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@dataclass
class Calibration:
    n: int
    rate: float
    mean_pred: float
    ece: float
    brier_skill: float
    bins: list[tuple[float, float, int, float, float]]  # lo, hi, n, pred, obs


def calibration(p: np.ndarray, y: np.ndarray, edges=None) -> Calibration:
    y = y.astype(float)
    edges = np.linspace(0, 1, 11) if edges is None else edges
    bins, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) | (hi >= 1.0) & (p <= hi))
        if m.any():
            bins.append((float(lo), float(hi), int(m.sum()),
                         float(p[m].mean()), float(y[m].mean())))
            ece += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    brier = float(np.mean((p - y) ** 2))
    ref = float(np.mean((y.mean() - y) ** 2))
    return Calibration(n=len(p), rate=float(y.mean()), mean_pred=float(p.mean()),
                       ece=float(ece), brier_skill=1 - brier / ref if ref > 0 else float("nan"),
                       bins=bins)


def _blocks(fr: Frames, idx: np.ndarray) -> np.ndarray:
    keys = np.char.add(fr.run[idx].astype(str),
                       np.floor(fr.t[idx] / BLOCK_S).astype(int).astype(str))
    return np.unique(keys, return_inverse=True)[1]


def bootstrap_ci(stat, fr: Frames, idx: np.ndarray, *arrays,
                 n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    """95 % interval of ``stat(*arrays)`` by resampling 2 s blocks."""
    if len(idx) < 2:
        return float("nan"), float("nan")
    block = _blocks(fr, idx)
    members = [np.flatnonzero(block == b) for b in range(block.max() + 1)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        sel = np.concatenate([members[b] for b in pick])
        v = stat(*(a[sel] for a in arrays))
        if np.isfinite(v):
            out.append(v)
    if len(out) < n_boot // 4:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _spearman(a, b) -> float:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")             # a constant ranks nothing
    return float(spearmanr(a, b).statistic)


# --------------------------------------------------------------------------
# predictions at the frames

@dataclass
class AtFrames:
    """Every layer, looked up at each frame's true cell and heading."""

    ok: np.ndarray             # (N,) frame landed on a scored cell
    row: np.ndarray            # (N,) row into result arrays, -1 if not ok
    hbin: np.ndarray           # (N,) heading bin
    flat: np.ndarray           # (N,) flat grid index, -1 off the grid
    layers: dict[str, np.ndarray]


LAYERS = ("track", "track_raw", "reloc", "combined", "n_eff", "expected_matched",
          "best_matches", "sigma_pos", "sigma_geom")


def lookup(result: Result, fr: Frames) -> AtFrames:
    grid = result.grid
    lut = np.full(grid.size, -1, dtype=np.int64)
    lut[result.cells] = np.arange(len(result.cells))
    flat = grid.sample_flat(np.nan_to_num(fr.xy, nan=1e9))
    row = np.where(flat >= 0, lut[np.maximum(flat, 0)], -1)
    heading = np.where(np.isfinite(fr.heading_deg), fr.heading_deg, 0.0)
    hbin = (np.floor(np.mod(heading, 360.0) / (360.0 / result.yaw_bins))
            .astype(int) % result.yaw_bins)
    ok = (row >= 0) & np.isfinite(fr.heading_deg) & np.isfinite(fr.xy).all(axis=1)
    layers = {}
    for name in LAYERS:
        arr = getattr(result, name)
        v = np.full(fr.n, np.nan)
        v[ok] = arr[row[ok], hbin[ok]]
        layers[name] = v
    return AtFrames(ok=ok, row=row, hbin=hbin, flat=flat, layers=layers)


# --------------------------------------------------------------------------
# what each layer should be compared against

def track_labels(fr: Frames, horizon_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Frames that were tracking, and whether the lock held for ``horizon_s``.

    Frames too close to the end of their run to know are left out.
    """
    sel, held = [], []
    for run in np.unique(fr.run):
        r = np.flatnonzero(fr.run == run)
        t, tr = fr.t[r], fr.tracked[r]
        # index of the first frame at or beyond t + horizon
        end = np.searchsorted(t, t + horizon_s)
        # running count of lost frames, to test "any loss in between" fast
        lost_cum = np.concatenate([[0], np.cumsum(~tr)])
        for k in range(len(r)):
            if not tr[k] or end[k] >= len(r):
                continue
            sel.append(r[k])
            held.append(lost_cum[end[k] + 1] - lost_cum[k] == 0)
    return np.array(sel, dtype=np.int64), np.array(held, dtype=bool)


@dataclass
class RelocTrials:
    idx: np.ndarray            # frames where stella was lost on the previous frame
    back: np.ndarray           # this frame has a pose again
    stable: np.ndarray         # ... and kept it for horizon_s
    one_frame: int             # comebacks lost again on the very next frame


def reloc_trials(fr: Frames, horizon_s: float) -> RelocTrials:
    """Every frame that was a relocalization attempt, and how it went.

    stella attempts relocalization on every frame while lost, so each lost
    frame is one failed attempt and the frame it comes back on is one success.
    """
    idx, back, stable, one = [], [], [], 0
    for run in np.unique(fr.run):
        r = np.flatnonzero(fr.run == run)
        tr, t = fr.tracked[r], fr.t[r]
        for k in range(1, len(r)):
            if tr[k - 1]:
                continue
            idx.append(r[k])
            back.append(bool(tr[k]))
            if tr[k]:
                end = np.searchsorted(t, t[k] + horizon_s)
                held = end < len(r) and bool(tr[k:end + 1].all())
                stable.append(held)
                if k + 1 < len(r) and not tr[k + 1]:
                    one += 1
            else:
                stable.append(False)
    return RelocTrials(np.array(idx, dtype=np.int64), np.array(back, dtype=bool),
                       np.array(stable, dtype=bool), one)


# --------------------------------------------------------------------------
# the checks

@dataclass
class LayerCheck:
    name: str
    question: str
    n: int
    auc: float = float("nan")
    auc_ci: tuple[float, float] = (float("nan"), float("nan"))
    rho: float = float("nan")
    rho_ci: tuple[float, float] = (float("nan"), float("nan"))
    cal: Calibration | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class RunsReport:
    frames: Frames
    n_on_grid: int
    convention_agreement: float | None
    checks: dict[str, LayerCheck]
    trials: RelocTrials


def check(fr: Frames, at: AtFrames, horizon_s: float,
          n_boot: int = N_BOOT,
          reference_cells: np.ndarray | None = None) -> RunsReport:
    """Score every layer against what the runs did.

    ``reference_cells`` is the cell list of a full-map heatmap built with the
    same grid -- the one the frames file's ``cell_row`` indexes into. Given
    it, cell and heading bin are cross-checked against the file, which
    catches a convention mismatch (axes, heading origin) before it can
    quietly corrupt every number below.
    """
    L = at.layers
    ok, gt, posed = at.ok, fr.gt, fr.posed
    out: dict[str, LayerCheck] = {}

    # --- landmarks: all frames with any pose and a lock ----------------------
    sel = np.flatnonzero(ok & posed & fr.tracked & (L["n_eff"] > 0))
    c = LayerCheck("landmarks", "does n_eff predict the landmarks stella matched?",
                   len(sel))
    if len(sel) > 2:
        c.rho = _spearman(L["n_eff"][sel], fr.n_matched[sel])
        c.rho_ci = bootstrap_ci(_spearman, fr, sel, L["n_eff"][sel],
                                fr.n_matched[sel], n_boot=n_boot)
        ratio = fr.n_matched[sel] / L["n_eff"][sel]
        c.extra["ratio_median"] = float(np.median(ratio))
        c.extra["ratio_iqr"] = (float(np.percentile(ratio, 25)),
                                float(np.percentile(ratio, 75)))
    g = np.flatnonzero(ok & gt)
    if len(g):
        c.auc = auc(L["n_eff"][g], fr.tracked[g])
        c.auc_ci = bootstrap_ci(auc, fr, g, L["n_eff"][g], fr.tracked[g], n_boot=n_boot)
    out["landmarks"] = c

    # --- tracking: already tracking -> holds for the horizon ------------------
    sel, held = track_labels(fr, horizon_s)
    keep = ok[sel] & gt[sel]
    sel, held = sel[keep], held[keep]
    c = LayerCheck("tracking",
                   f"already tracking: does the lock hold for {horizon_s:g} s?",
                   len(sel))
    if len(sel):
        c.auc = auc(L["track"][sel], held)
        c.auc_ci = bootstrap_ci(auc, fr, sel, L["track"][sel], held, n_boot=n_boot)
        c.cal = calibration(L["track"][sel], held)
        c.extra["losses"] = int((~held).sum())
    # also the reading in the first real-run check, for comparison
    g = np.flatnonzero(ok & gt)
    if len(g):
        c.extra["unconditional_auc"] = auc(L["track"][g], fr.tracked[g])
        c.extra["unconditional_cal"] = calibration(L["track"][g], fr.tracked[g])
    out["tracking"] = c

    # --- relocalization: lost -> back on this frame ---------------------------
    tri = reloc_trials(fr, horizon_s)
    keep = ok[tri.idx] & gt[tri.idx]
    idx, back = tri.idx[keep], tri.back[keep]
    c = LayerCheck("relocalization", "while lost: does this frame bring it back?",
                   len(idx))
    if len(idx):
        c.auc = auc(L["reloc"][idx], back)
        c.auc_ci = bootstrap_ci(auc, fr, idx, L["reloc"][idx], back, n_boot=n_boot)
        c.cal = calibration(L["reloc"][idx], back)
        c.extra["comebacks"] = int(back.sum())
        c.extra["stable"] = int(tri.stable[keep].sum())
        c.extra["share_rated_half"] = float((L["reloc"][idx][~back] >= 0.5).mean()) \
            if (~back).any() else float("nan")
        c.extra["best_matches_auc"] = auc(L["best_matches"][idx], back)
    out["relocalization"] = c

    # --- position: tracked, ground truth, error known -------------------------
    sel = np.flatnonzero(ok & gt & fr.tracked & np.isfinite(fr.error_m)
                         & (fr.error_m > 0))
    c = LayerCheck("position", "is sigma_pos the size of the real error?", len(sel))
    if len(sel) > 2:
        s, e = L["sigma_pos"][sel], fr.error_m[sel]
        c.rho = _spearman(s, e)
        c.rho_ci = bootstrap_ci(_spearman, fr, sel, s, e, n_boot=n_boot)
        c.extra["sigma_median_m"] = float(np.median(s))
        c.extra["error_median_m"] = float(np.median(e))
        c.extra["within_95_bound"] = float((e <= MAXWELL_95 * s).mean())
    out["position"] = c

    agree = None
    has = ok & (fr.their_cell >= 0) & (fr.their_bin >= 0)
    if reference_cells is not None and has.any():
        theirs = np.full(fr.n, -2, dtype=np.int64)
        valid = has & (fr.their_cell < len(reference_cells))
        theirs[valid] = reference_cells[fr.their_cell[valid]]
        agree = float(((theirs == at.flat) & (fr.their_bin == at.hbin))[has].mean())

    return RunsReport(frames=fr, n_on_grid=int(ok.sum()),
                      convention_agreement=agree, checks=out, trials=tri)


# --------------------------------------------------------------------------
# fitting

@dataclass
class Fit:
    matched_per_neff: float | None = None
    calib_a: float | None = None
    calib_b: float | None = None
    bow_match_yield: float | None = None
    yield_ci: tuple[float, float] | None = None
    sigma_scale: float | None = None
    sigma_floor_m: float | None = None
    notes: list[str] = field(default_factory=list)

    def apply(self, cfg: Config) -> Config:
        c = copy.deepcopy(cfg)
        fitted = list(c.calibration.fitted)
        def put(section, key, value, tag):
            if value is not None:
                setattr(getattr(c, section), key, float(value))
                if tag not in fitted:
                    fitted.append(tag)
        put("calibration", "matched_per_neff", self.matched_per_neff, "landmarks")
        put("tracking", "calib_a", self.calib_a, "tracking")
        put("tracking", "calib_b", self.calib_b, "tracking")
        put("relocalization", "bow_match_yield", self.bow_match_yield, "relocalization")
        put("tracking", "sigma_scale", self.sigma_scale, "position")
        put("tracking", "sigma_floor_m", self.sigma_floor_m, "position")
        c.calibration.fitted = fitted
        return c


def _platt(raw: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    q = np.clip(raw, 1e-4, 1 - 1e-4)
    z = np.log(q / (1 - q))
    y = y.astype(float)

    def nll(ab):
        a, b = ab
        s = a * z + b
        # log(1 + e^s) computed stably
        return (np.sum(np.logaddexp(0, s) - y * s)
                + PLATT_PRIOR * ((a - 1) ** 2 + b ** 2))

    res = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6})
    return float(res.x[0]), float(res.x[1])


def _maxwell_nll(err: np.ndarray, sigma: np.ndarray) -> float:
    s = sigma / np.sqrt(3.0)
    return float(-np.sum(0.5 * np.log(2 / np.pi) + 2 * np.log(err)
                         - 3 * np.log(s) - err ** 2 / (2 * s ** 2)))


def fit(model: MapModel, grid: Grid, occ: Occupancy, vis: Visibility,
        result: Result, fr: Frames, at: AtFrames, cfg: Config,
        height: float, horizon_s: float, n_boot: int = N_BOOT) -> Fit:
    """Fit the parameters real images decide, one layer at a time."""
    f = Fit()
    L, ok, gt = at.layers, at.ok, fr.gt

    # 1. landmarks: how many stella matches per predicted one
    sel = ok & fr.posed & fr.tracked & (L["n_eff"] > 0)
    if sel.sum() >= 30:
        f.matched_per_neff = float(np.median(fr.n_matched[sel] / L["n_eff"][sel]))
    else:
        f.notes.append("landmarks: fewer than 30 tracked frames, not fitted")

    # 2. tracking: Platt on the raw score, frames already tracking
    tsel, held = track_labels(fr, horizon_s)
    keep = ok[tsel] & gt[tsel]
    tsel, held = tsel[keep], held[keep]
    if len(tsel) >= 50 and 0 < held.sum() < len(held):
        f.calib_a, f.calib_b = _platt(L["track_raw"][tsel], held)
        n_loss = int((~held).sum())
        if n_loss < 30:
            f.notes.append(f"tracking: only {n_loss} frames lost the lock; "
                           f"the calibration leans on its prior")
    else:
        f.notes.append("tracking: not enough frames, or no losses, to calibrate")

    # 3. relocalization: per-frame likelihood of the comebacks, over yields
    tri = reloc_trials(fr, horizon_s)
    keep = ok[tri.idx] & gt[tri.idx]
    ridx, back = tri.idx[keep], tri.back[keep]
    if len(ridx) >= 30 and back.any():
        cells = np.unique(at.row[ridx])
        need = np.isin(np.arange(len(result.cells)), cells)
        sub_vis = vis.subset(need)
        sub_rows = np.full(len(result.cells), -1)
        sub_rows[need] = np.arange(need.sum())
        base = copy.deepcopy(cfg)
        if f.matched_per_neff is not None:
            base.calibration.matched_per_neff = f.matched_per_neff
        preds = np.empty((len(YIELD_GRID), len(ridx)))
        for k, yld in enumerate(YIELD_GRID):
            c = copy.deepcopy(base)
            c.relocalization.bow_match_yield = float(yld)
            r = scorers.evaluate(model, grid, occ, sub_vis, height=height,
                                 yaw_bins=result.yaw_bins, cfg=c)
            preds[k] = r.reloc[sub_rows[at.row[ridx]], at.hbin[ridx]]
        p = np.clip(preds, 1e-6, 1 - 1e-6)
        ll = back * np.log(p) + (~back) * np.log(1 - p)        # (yields, trials)
        k_best = int(np.argmax(ll.sum(axis=1)))
        f.bow_match_yield = float(YIELD_GRID[k_best])
        if k_best in (0, len(YIELD_GRID) - 1):
            f.notes.append(f"relocalization: the best yield is at the edge of the "
                           f"range searched ({YIELD_GRID[0]:g}-{YIELD_GRID[-1]:g}); "
                           f"the model cannot match the observed rate")
        # interval: resample 2 s blocks, refit
        block = _blocks(fr, ridx)
        members = [np.flatnonzero(block == b) for b in range(block.max() + 1)]
        rng = np.random.default_rng(1)
        picks = []
        for _ in range(n_boot):
            pick = np.concatenate([members[b] for b in
                                   rng.integers(0, len(members), len(members))])
            picks.append(YIELD_GRID[np.argmax(ll[:, pick].sum(axis=1))])
        f.yield_ci = (float(np.percentile(picks, 2.5)),
                      float(np.percentile(picks, 97.5)))
        if back.sum() < 20:
            f.notes.append(f"relocalization: only {int(back.sum())} comebacks "
                           f"to fit on; treat the yield as a first estimate")
    else:
        f.notes.append("relocalization: not enough lost frames or no comeback")

    # 4. position: scale on geometry, plus a floor, by 3D error likelihood
    psel = ok & gt & fr.tracked & np.isfinite(fr.error_m) & (fr.error_m > 0)
    if psel.sum() >= 50:
        geom = L["sigma_geom"][psel] / max(cfg.tracking.sigma_scale, 1e-9)
        err = fr.error_m[psel]
        best = (np.inf, None, None)
        scales = np.concatenate([[0.0], np.geomspace(0.05, 8.0, 41)])
        for k in scales:
            for fl in np.linspace(0.0, 0.05, 101):
                sig = np.sqrt((k * geom) ** 2 + fl ** 2)
                v = _maxwell_nll(err, np.maximum(sig, 1e-5))
                if v < best[0]:
                    best = (v, k, fl)
        f.sigma_scale, f.sigma_floor_m = float(best[1]), float(best[2])
        if best[1] == 0.0:
            f.notes.append("position: the best fit drops the geometric term "
                           "entirely -- real error here does not follow the "
                           "landmark geometry, so sigma_pos becomes a constant "
                           "of the right size rather than a map of where error "
                           "is larger")
        f.notes.append("position: errors below ~0.5 cm are within the ground "
                       "truth's own uncertainty")
    else:
        f.notes.append("position: fewer than 50 frames with a known error")
    return f


# --------------------------------------------------------------------------
# reporting

def _ci(ci) -> str:
    return f"[{ci[0]:.2f} to {ci[1]:.2f}]" if np.all(np.isfinite(ci)) else "[n/a]"


def describe(rep: RunsReport, title: str) -> str:
    c = rep.checks
    lines = [f"  {title}"]

    lm = c["landmarks"]
    lines.append(
        f"    landmarks       rank corr {lm.rho:+.2f} {_ci(lm.rho_ci)}; matched = "
        f"{lm.extra.get('ratio_median', float('nan')):.2f} x n_eff; "
        f"tracked-vs-lost AUC {lm.auc:.2f} {_ci(lm.auc_ci)}  ({lm.n} frames)")

    tr = c["tracking"]
    if tr.cal:
        lines.append(
            f"    tracking        {tr.question}  {tr.n} frames, "
            f"{tr.extra['losses']} did not")
        lines.append(
            f"                    AUC {tr.auc:.2f} {_ci(tr.auc_ci)}; predicted "
            f"{tr.cal.mean_pred:.2f} vs held {tr.cal.rate:.2f}; ECE {tr.cal.ece:.3f}; "
            f"Brier skill {tr.cal.brier_skill:+.2f}")
        for lo, hi, n, p, o in tr.cal.bins:
            if n >= 5:
                lines.append(f"                      predicted {lo:.1f}-{hi:.1f}: "
                             f"{n:5d} frames, mean {p:.2f}, held {o:.2f}")
        u = tr.extra.get("unconditional_cal")
        if u:
            lines.append(
                f"                    (asked of every frame instead: AUC "
                f"{tr.extra['unconditional_auc']:.2f}, predicted {u.mean_pred:.2f} "
                f"vs tracked {u.rate:.2f} -- startup frames that had not yet "
                f"relocalized count against tracking there)")

    rl = c["relocalization"]
    if rl.cal:
        lines.append(
            f"    relocalization  {rl.question}  {rl.n} attempts, "
            f"{rl.extra['comebacks']} came back ({rl.extra['stable']} held)")
        lines.append(
            f"                    AUC {rl.auc:.2f} {_ci(rl.auc_ci)}; predicted "
            f"{rl.cal.mean_pred:.3f} vs observed {rl.cal.rate:.3f} per attempt; "
            f"ECE {rl.cal.ece:.3f}; Brier skill {rl.cal.brier_skill:+.2f}; "
            f"{100 * rl.extra['share_rated_half']:.0f}% of failed attempts "
            f"rated >= 0.5")

    ps = c["position"]
    if ps.n > 2:
        corr = (f"rank corr {ps.rho:+.2f} {_ci(ps.rho_ci)}" if np.isfinite(ps.rho)
                else "constant sigma, so no ranking")
        lines.append(
            f"    position        {corr}; "
            f"sigma {100 * ps.extra['sigma_median_m']:.1f} cm vs error "
            f"{100 * ps.extra['error_median_m']:.1f} cm (medians); "
            f"{100 * ps.extra['within_95_bound']:.0f}% within the 95% bound "
            f"(target 95%)")
    return "\n".join(lines)


def describe_header(rep: RunsReport, model: MapModel, runs: list[str]) -> str:
    fr = rep.frames
    lines = [
        f"{model.raw.path.name} against {len(runs)} run(s): {', '.join(runs)}",
        f"  {fr.n} frames: {int(fr.gt.sum())} with ground truth, "
        f"{int((fr.source == 'slam').sum())} placed by stella; "
        f"{rep.n_on_grid} land on a scored cell",
    ]
    if rep.convention_agreement is not None:
        lines.append(
            f"  cell and heading bin agree with the frames file for "
            f"{100 * rep.convention_agreement:.1f}% of frames "
            f"(meaningful only if the grid and headings are the same as theirs)")
    t = rep.trials
    back = int(t.back.sum())
    if back:
        lines.append(
            f"  {back} relocalizations in all; {t.one_frame} were lost again on "
            f"the very next frame (stella has no motion model after a "
            f"relocalization and falls back to BoW matching)")
    return "\n".join(lines)


def describe_fit(f: Fit, cfg: Config) -> str:
    lines = ["  fitted:"]
    if f.matched_per_neff is not None:
        lines.append(f"    calibration.matched_per_neff  "
                     f"{cfg.calibration.matched_per_neff:g} -> {f.matched_per_neff:.3f}")
    if f.calib_a is not None:
        lines.append(f"    tracking.calib_a / calib_b    "
                     f"{cfg.tracking.calib_a:g} / {cfg.tracking.calib_b:g} -> "
                     f"{f.calib_a:.3f} / {f.calib_b:+.3f}")
    if f.bow_match_yield is not None:
        lines.append(f"    relocalization.bow_match_yield "
                     f"{cfg.relocalization.bow_match_yield:g} -> "
                     f"{f.bow_match_yield:.3f}  (95% {f.yield_ci[0]:.3f} to "
                     f"{f.yield_ci[1]:.3f})")
    if f.sigma_scale is not None:
        lines.append(f"    tracking.sigma_scale / floor  "
                     f"{cfg.tracking.sigma_scale:g} / {100 * cfg.tracking.sigma_floor_m:g} cm "
                     f"-> {f.sigma_scale:.2f} / {100 * f.sigma_floor_m:.2f} cm")
    for n in f.notes:
        lines.append(f"    note: {n}")
    return "\n".join(lines)


def provenance(model: MapModel, runs: list[str]) -> str:
    return (f"{date.today().isoformat()}: {model.raw.path.name} against "
            f"{', '.join(runs)}")
