"""Write results out: raw arrays for further work, and a viewer to look at.

The viewer is one self-contained HTML file with the data embedded, so it can
be opened from disk, e-mailed, or dropped in a report without a server or a
build step.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .mapmodel import MapModel
from .occupancy import Occupancy
from .config import DEFAULT, Config
from .scorers import Result

TEMPLATE = Path(__file__).with_name("viewer_template.html")


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _quantise(values: np.ndarray, lo: float, hi: float, bits: int
              ) -> tuple[str, float]:
    """Pack a float layer into 8 or 16 bit integers for transport."""
    top = (1 << bits) - 1
    scale = top / max(hi - lo, 1e-9)
    q = np.clip((values - lo) * scale, 0, top)
    dtype = np.uint16 if bits == 16 else np.uint8
    return _b64(q.astype(dtype)), float(scale)


def write(out_dir: Path, model: MapModel, occ: Occupancy, tracking: Result,
          params: dict, plan=None, cfg: Config | None = None) -> dict[str, Path]:
    """Write ``.npz``, a JSON sidecar, and the viewer. Returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = model.raw.path.stem
    grid = tracking.grid

    npz = out_dir / f"{stem}.npz"
    if plan is not None and plan.suggestions:
        meta_suggest = np.array([[s.xy[0], s.xy[1]] for s in plan.suggestions])
    else:
        meta_suggest = np.zeros((0, 2))
    np.savez_compressed(
        npz,
        suggestions=meta_suggest,
        occupancy=occ.state,
        log_odds=occ.log_odds,
        cells=tracking.cells.astype(np.int32),
        track=tracking.track,
        reloc=tracking.reloc,
        combined=tracking.combined,
        best_matches=tracking.best_matches,
        n_eff=tracking.n_eff,
        n_raw=tracking.n_raw,
        sigma_pos=tracking.sigma_pos,
        # Added in 0.3. Existing names keep their meaning; readers that look
        # layers up by name (the Qualisys alignment app does) are unaffected.
        expected_matched=tracking.expected_matched,
        track_raw=tracking.track_raw,
        sigma_geom=tracking.sigma_geom,
        reloc_within=tracking.reloc_within,
        reloc_time_s=tracking.reloc_time_s,
        stations=model.stations.center,
        origin=grid.origin,
        res=np.array([grid.res]),
        shape=np.array([grid.nx, grid.ny]),
    )

    meta = {
        "map": str(model.raw.path),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "grid": {"origin": grid.origin.tolist(), "res": grid.res,
                 "nx": grid.nx, "ny": grid.ny},
        "yaw_bins": tracking.yaw_bins,
        "sample_height_m": tracking.height,
        "up_axis": model.up_axis,
        "stations": model.stations.center.tolist(),
        "params": params,
        "stats": {
            "cells_scored": int(len(tracking.cells)),
            "n_eff_median": float(np.median(tracking.n_eff)),
            "track_mean": float(tracking.track.mean()),
            "reloc_mean": float(tracking.reloc.mean()),
            "combined_mean": float(tracking.combined.mean()),
            "combine_mode": tracking.combine_mode,
            "usable_any_heading": float((tracking.combined.max(axis=1) > 0.5).mean()),
            "usable_every_heading": float((tracking.combined.min(axis=1) > 0.5).mean()),
        },
    }
    sidecar = out_dir / f"{stem}.json"
    sidecar.write_text(json.dumps(meta, indent=2))

    html = out_dir / f"{stem}.html"
    html.write_text(_render(model, occ, tracking, meta, plan, cfg))
    return {"npz": npz, "json": sidecar, "html": html}


def _station_yaws(model: MapModel) -> list[list[float]]:
    """The directions each station actually looked, for the map overlay."""
    raw = model.raw
    out: list[list[float]] = []
    for rows in model.stations.kf_rows:
        # optical axis of each keyframe, as a compass heading
        yaws = np.degrees(model.heading(raw.kf_rot_cw[rows, 2, :]))
        out.append(sorted({round(float(y), 1) for y in yaws}))
    return out


def _render(model: MapModel, occ: Occupancy, tr: Result, meta: dict,
            plan=None, cfg: Config | None = None) -> str:
    conf = cfg or DEFAULT
    marginal = conf.tracking.n_tracked_marginal
    min_bow = conf.relocalization.min_bow_matches
    grid = tr.grid
    cal = conf.calibration
    fitted = set(cal.fitted)
    find_s = conf.relocalization.find_within_s

    def status(*tags: str) -> tuple[bool, str]:
        if all(t in fitted for t in tags):
            return True, f"Calibrated on real runs ({cal.source})."
        missing = ", ".join(t for t in tags if t not in fitted)
        return False, (f"Model default for {missing}: not yet checked against "
                       f"a real run. `validate-runs --fit` calibrates it.")

    em_hi = float(max(np.percentile(tr.expected_matched, 99), 10.0))
    match_hi = float(max(np.percentile(tr.best_matches, 99), 40.0))
    comb_b, comb_s = _quantise(tr.combined, 0.0, 1.0, 8)
    track_b, track_s = _quantise(tr.track, 0.0, 1.0, 8)
    em_b, em_s = _quantise(tr.expected_matched, 0.0, em_hi, 16)
    rt_b, rt_s = _quantise(np.log10(np.clip(tr.reloc_time_s, 0.01, 3600.0)),
                           -2.0, np.log10(3600.0), 16)
    match_b, match_s = _quantise(tr.best_matches, 0.0, match_hi, 16)
    sig_b, sig_s = _quantise(tr.sigma_pos * 100.0, 0.0, 100.0, 16)

    combine_help = {
        "product": "both have to work",
        "min": "the weaker of the two",
        "sum": "the average of the two",
    }[tr.combine_mode]

    def layer(tags, **kw):
        ok, note = status(*tags)
        kw.update(calibrated=ok, status=note)
        return kw

    layers = {
        "track": layer(
            ("tracking",),
            label="Keeps tracking",
            data=track_b, scale=track_s, bits=8,
            lo=0.0, hi=1.0, higherIsBetter=True, log=False, fmt="ratio",
            help=f"Chance a device that is already tracking keeps its lock for "
                 f"the next {conf.tracking.horizon_s:g} s here. The best-"
                 f"validated layer: start with this one.",
            **{"def": "Once a device knows where it is, can it hold on while it "
                      "moves? That needs enough matchable map points in view, "
                      "spread well enough to fix a pose. 1 is certain, 0 is "
                      "hopeless. Real runs agree with this layer best."}),
        "expected_matched": layer(
            ("landmarks",),
            label="Expected matched landmarks",
            data=em_b, scale=em_s, bits=16,
            lo=1.0, hi=em_hi, higherIsBetter=True, log=True, fmt="count",
            help=f"How many map points stella should match in a frame here. "
                 f"Below {marginal:.0f} tracking is at risk; stella gives up "
                 f"below 20.",
            **{"def": "A real count, in the same units stella reports: the "
                      "predicted trackable points times "
                      f"{cal.matched_per_neff:.2f}, the share stella actually "
                      "matched on real runs. Compare against 20 (lost), "
                      f"{marginal:.0f} (marginal), 100 (solid)."}),
        "reloc_time": layer(
            ("relocalization",),
            label="Time to find itself",
            data=rt_b, scale=rt_s, bits=16,
            lo=0.01, hi=3600.0, higherIsBetter=False, log=True, fmt="seconds",
            transform="pow10", offset=-2.0, averageAs="rate",
            help="Expected wait, starting from nothing, until stella works out "
                 "where it is. It tries on every frame.",
            **{"def": "Relocalization: finding yourself cold, the hard part and "
                      "what fails when a device will not start up. stella tries "
                      "on every frame; this is one over the chance per try. "
                      "First real runs: it cannot yet tell good places from bad "
                      "ones, only how long it takes on average."}),
        "combined": layer(
            ("tracking", "relocalization"),
            label=f"Finds itself within {find_s:g} s and keeps tracking",
            data=comb_b, scale=comb_s, bits=8,
            lo=0.0, hi=1.0, higherIsBetter=True, log=False, fmt="ratio",
            help=f"Relocalizing within {find_s:g} s and then keeping the lock: "
                 f"{combine_help}.",
            **{"def": "Both halves together. Where this is dark, the two layers "
                      "above say which half fails. Until relocalization is "
                      "calibrated on real runs, read them side by side rather "
                      "than relying on this one."}),
        "sigma_pos": layer(
            ("position",),
            label="Position uncertainty",
            data=sig_b, scale=sig_s, bits=16,
            lo=0.5, hi=50.0, higherIsBetter=False, log=True, fmt="cm",
            help="How far off the position is likely to be, in centimetres. "
                 "Lower is better.",
            **{"def": "Geometry of the visible points, plus a floor for errors "
                      "that no number of points averages away -- chiefly the "
                      "map's own offset from the world. On the first real runs "
                      "the size is right, but it does not yet predict where "
                      "error is larger."}),
        "best_matches": layer(
            ("relocalization",),
            label="BoW matches to best view",
            data=match_b, scale=match_s, bits=16,
            lo=1.0, hi=match_hi, higherIsBetter=True, log=True, fmt="count",
            help=f"Expected bag-of-words matches against the most similar "
                 f"stored view. stella needs {min_bow:.0f} to try a pose.",
            **{"def": "The first and hardest relocalization gate: matching a "
                      "real image's descriptors against the map's. After that "
                      f"stella needs {conf.relocalization.min_valid_obs:.0f} "
                      "landmarks to survive pose refinement."}),
    }

    spacing = model.stations.spacing()
    stats = [
        ["scan positions", f"{model.stations.n}",
         "Distinct places the scanner stood. Everything the map knows was seen "
         "from one of these."],
        ["station spacing", f"{np.median(spacing):.1f} m" if model.stations.n > 1 else "—",
         "Median distance between neighbouring scan positions."],
        ["landmarks", f"{int((model.weight > 0).sum())}",
         "Usable 3D map points, after removing duplicates and defects."],
        ["sample height", f"{tr.height:.2f} m",
         "Height above the world origin at which the virtual camera was flown. "
         "Matches the height the scanner was at."],
        ["grid", f"{grid.res * 100:.0f} cm",
         "Size of one cell in the heatmap."],
        ["headings", f"{tr.yaw_bins}",
         f"Every cell was scored facing {tr.yaw_bins} different directions, "
         f"{360 // tr.yaw_bins} degrees apart."],
        ["usable somewhere", f"{100 * meta['stats']['usable_any_heading']:.0f} %",
         "Share of open floor where at least one heading scores above 0.5."],
        ["usable facing anywhere", f"{100 * meta['stats']['usable_every_heading']:.0f} %",
         "Share of open floor that works no matter which way you turn. This is "
         "the honest number for a device that can look in any direction."],
        ["combining", tr.combine_mode,
         "How the relocalization and tracking layers are merged. 'product' "
         "means both have to work."],
        ["calibrated on real runs", ", ".join(cal.fitted) or "nothing yet",
         "Layers whose numbers were fitted to real localization runs. The rest "
         "are model defaults and are marked as such."],
    ]

    caveats = (
        "Grey areas were never scanned, so nothing is claimed about them. "
        "Layers marked <b>uncalibrated</b> have not been checked against a real "
        "run yet. "
        "Relocalization is modelled from how much structure a stored view and "
        "your view share; it cannot see <b>look-alike places</b>, so two "
        "corridors that are genuinely identical will both score well when in "
        "reality a device would confuse them."
    )

    payload = {
        "name": model.raw.path.name,
        "subtitle": f"{grid.nx}x{grid.ny} cells at {grid.res * 100:.0f} cm  ·  "
                    f"up={model.up_axis}  ·  {meta['generated'][:10]}",
        "origin": grid.origin.tolist(),
        "res": grid.res,
        "nx": grid.nx, "ny": grid.ny,
        "yawBins": tr.yaw_bins,
        "defaultLayer": "track",
        "fps": float(model.camera.fps or 30.0),
        "occ": _b64(occ.state.astype(np.uint8)),
        "cells": _b64(tr.cells.astype(np.int32)),
        "layers": layers,
        "stations": model.ground(model.stations.center).tolist(),
        "stationYaws": _station_yaws(model),
        "stats": stats,
        "caveats": caveats,
        "suggestions": [
            {"rank": s.rank, "x": s.xy[0], "y": s.xy[1],
             "cells": s.cells_helped,
             "before": round(s.mean_before, 2), "after": round(s.mean_after, 2)}
            for s in (plan.suggestions if plan else [])
        ],
        "suggestNote": (
            f"Best heading coverage would go from "
            f"{100 * plan.coverage_before:.0f}% to {100 * plan.coverage_after:.0f}%."
            + ("" if plan.reachable >= plan.target else
               f" Even at an existing station this map only reaches "
               f"{plan.reachable:.2f}, so more stations will not clear "
               f"{plan.target:.2f} — the limit is capture quality, not density.")
        ) if plan and plan.suggestions else "",
    }

    data = json.dumps(payload)
    template = TEMPLATE.read_text()
    return (template
            .replace("__TITLE__", f"{model.raw.path.stem} — localizability")
            .replace("__DATA__", data))
