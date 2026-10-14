"""Render a layer to a PNG that stands on its own in a report.

The viewer is for exploring; this is for pasting into a document, where
nobody can hover a cell to find out what the colours mean.  So the furniture
is part of the image: a title, a scale bar in metres, a colour ramp with real
values on it, the scan positions, and a footer recording the settings the
figure was made with.  A screenshot of the viewer has none of that, and a
year later nobody can tell what they are looking at.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .mapmodel import MapModel
from .occupancy import FREE, OCCUPIED, Occupancy
from .scorers import Result

VIRIDIS = np.array([
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
    [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37],
], dtype=np.float64)

PAPER = (247, 248, 246)
INK = (18, 26, 27)
INK_2 = (90, 104, 102)
RULE = (206, 214, 209)
GROUND = (232, 236, 232)
OPEN = (255, 255, 255)
WALL = (24, 32, 33)
SUGGEST = (255, 138, 43)

MARGIN = 26
LEGEND_H = 54
HEADER_H = 46
FOOTER_H = 34


def _font(size: int, bold: bool = False):
    names = (
        ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"] if bold
        else ["DejaVuSans.ttf", "LiberationSans-Regular.ttf"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)    # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _viridis(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0) * (len(VIRIDIS) - 1)
    i = np.minimum(t.astype(int), len(VIRIDIS) - 2)
    f = (t - i)[..., None]
    return VIRIDIS[i] * (1 - f) + VIRIDIS[i + 1] * f


def _nice_scale(res: float, px_per_cell: int, target_px: int = 150) -> float:
    """A round number of metres that is roughly ``target_px`` wide."""
    want = target_px * res / px_per_cell
    for step in (0.5, 1, 2, 5, 10, 20, 50, 100):
        if step >= want:
            return float(step)
    return 100.0


def layer_png(path: Path, model: MapModel, occ: Occupancy, result: Result,
              layer: str = "combined", aggregate: str = "mean",
              scale: int = 4, plan=None, params: dict | None = None) -> Path:
    """Write one layer as a self-describing PNG."""
    values, lo, hi, label, unit, higher_better, log = _layer_spec(result, layer)
    grid = result.grid

    # "max" is the best heading and "min" the worst, whichever way the layer
    # points. Waiting times average through their rate, or one hopeless
    # heading swamps the rest.
    best = np.max if higher_better else np.min
    worst = np.min if higher_better else np.max
    if aggregate == "mean":
        if unit == "seconds":
            v = 1.0 / np.mean(1.0 / np.maximum(values, 1e-12), axis=1)
        else:
            v = values.mean(axis=1)
    elif aggregate == "max":
        v = best(values, axis=1)
    elif aggregate == "min":
        v = worst(values, axis=1)
    else:
        raise ValueError(f"unknown aggregate {aggregate!r}")

    # --- the map itself, at grid resolution --------------------------------
    rgb = np.zeros((grid.nx, grid.ny, 3), dtype=np.float64)
    rgb[:] = GROUND
    rgb[occ.state == FREE] = OPEN
    ix, iy = np.divmod(result.cells, grid.ny)
    if log:
        t = (np.log10(np.maximum(v, lo)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    else:
        t = (v - lo) / (hi - lo)
    rgb[ix, iy] = _viridis(t if higher_better else 1.0 - t)
    rgb[occ.state == OCCUPIED] = WALL

    # rows top-down, north up
    img = Image.fromarray(
        np.transpose(rgb, (1, 0, 2))[::-1].astype(np.uint8), "RGB"
    ).resize((grid.nx * scale, grid.ny * scale), Image.NEAREST)

    W = img.width + 2 * MARGIN
    H = img.height + 2 * MARGIN + HEADER_H + LEGEND_H + FOOTER_H
    canvas = Image.new("RGB", (W, H), PAPER)
    top = MARGIN + HEADER_H
    canvas.paste(img, (MARGIN, top))
    d = ImageDraw.Draw(canvas)
    d.rectangle([MARGIN, top, MARGIN + img.width - 1, top + img.height - 1],
                outline=RULE)

    def world_to_px(x: float, y: float) -> tuple[float, float]:
        return (
            MARGIN + (x - grid.origin[0]) / grid.res * scale,
            top + img.height - (y - grid.origin[1]) / grid.res * scale,
        )

    # --- scan positions, and any recommended ones --------------------------
    for cx, cy in model.ground(model.stations.center):
        px, py = world_to_px(cx, cy)
        d.ellipse([px - 4, py - 4, px + 4, py + 4], outline=(255, 255, 255), width=3)
        d.ellipse([px - 4, py - 4, px + 4, py + 4], outline=INK, width=1)
    for s in (plan.suggestions if plan else []):
        px, py = world_to_px(*s.xy)
        d.polygon([(px, py - 6), (px + 6, py), (px, py + 6), (px - 6, py)],
                  fill=SUGGEST, outline=(255, 255, 255))
        tag = str(s.rank)
        d.text((px + 9, py - 7), tag, font=_font(12, True), fill=(255, 255, 255),
               stroke_width=3, stroke_fill=(255, 255, 255))
        d.text((px + 9, py - 7), tag, font=_font(12, True), fill=INK)

    # --- header ------------------------------------------------------------
    d.text((MARGIN, MARGIN - 6), f"{model.raw.path.stem} — {label}",
           font=_font(17, True), fill=INK)
    d.text((MARGIN, MARGIN + 15),
           f"{unit}, {aggregate} over {result.yaw_bins} headings"
           f"   ·   camera height {result.height:.2f} m"
           f"   ·   {model.stations.n} scan positions",
           font=_font(11), fill=INK_2)

    # --- scale bar, drawn over the map -------------------------------------
    metres = _nice_scale(grid.res, scale)
    bar = metres / grid.res * scale
    bx, by = MARGIN + 12, top + img.height - 14
    d.rectangle([bx - 4, by - 15, bx + bar + 4, by + 5], fill=(255, 255, 255))
    d.line([(bx, by), (bx + bar, by)], fill=INK, width=2)
    for x in (bx, bx + bar):
        d.line([(x, by - 4), (x, by + 2)], fill=INK, width=2)
    label_m = f"{metres:g} m"
    d.text((bx + bar / 2 - len(label_m) * 3, by - 15), label_m,
           font=_font(11), fill=INK)

    # --- legend ------------------------------------------------------------
    ly = top + img.height + 18
    lw = min(300, img.width)
    ramp = np.linspace(0, 1, lw)
    strip = _viridis(ramp if higher_better else 1 - ramp)
    strip = np.repeat(strip[None, :, :], 13, axis=0).astype(np.uint8)
    canvas.paste(Image.fromarray(strip, "RGB"), (MARGIN, ly))
    d.rectangle([MARGIN, ly, MARGIN + lw - 1, ly + 12], outline=RULE)
    if unit == "seconds":
        fmt = lambda x: f"{x:.2g} s" if x < 120 else f"{x / 60:.2g} min"  # noqa: E731
    else:
        fmt = (lambda x: f"{x:.0f}") if hi > 20 else (lambda x: f"{x:.3g}")  # noqa: E731
    for frac in (0.0, 0.5, 1.0):
        val = lo * (hi / lo) ** frac if log else lo + frac * (hi - lo)
        tx = MARGIN + frac * lw
        d.text((tx - (0 if frac == 0 else 12 if frac < 1 else 24), ly + 16),
               fmt(val), font=_font(11), fill=INK_2)
    d.text((MARGIN + lw + 14, ly),
           "higher is better" if higher_better else "lower is better",
           font=_font(11), fill=INK_2)

    legend_x = MARGIN + lw + 120
    d.ellipse([legend_x, ly, legend_x + 9, ly + 9], outline=INK, width=2)
    d.text((legend_x + 15, ly - 2), "scan position", font=_font(11), fill=INK_2)
    if plan and plan.suggestions:
        d.polygon([(legend_x + 5, ly + 15), (legend_x + 11, ly + 21),
                   (legend_x + 5, ly + 27), (legend_x - 1, ly + 21)],
                  fill=SUGGEST, outline=INK)
        d.text((legend_x + 15, ly + 14), "suggested next scan",
               font=_font(11), fill=INK_2)

    # --- footer: what this figure was made with ----------------------------
    bits = [f"grid {grid.res * 100:.0f} cm", f"up {model.up_axis}"]
    if params:
        for key in ("combine", "occlusion", "yaw_bins"):
            if key in params:
                bits.append(f"{key} {params[key]}")
    d.line([(MARGIN, H - FOOTER_H + 2), (W - MARGIN, H - FOOTER_H + 2)], fill=RULE)
    d.text((MARGIN, H - FOOTER_H + 10),
           "slammap2heatmap  ·  " + "  ·  ".join(bits),
           font=_font(10), fill=INK_2)

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def _layer_spec(r: Result, layer: str):
    """(values, lo, hi, label, unit, higher_is_better, log)"""
    if layer == "combined":
        return (r.combined, 0.0, 1.0, "Finds itself and keeps tracking",
                "probability", True, False)
    if layer == "track":
        return r.track, 0.0, 1.0, "Keeps tracking", "probability", True, False
    if layer == "expected_matched":
        hi = float(max(np.percentile(r.expected_matched, 99), 10.0))
        return (r.expected_matched, 1.0, hi, "Expected matched landmarks",
                "landmarks per frame", True, True)
    if layer == "reloc_time":
        return (r.reloc_time_s, 0.1, 600.0, "Time to find itself", "seconds",
                False, True)
    if layer == "reloc":
        return r.reloc, 0.0, 1.0, "Relocalization", "probability", True, False
    if layer == "n_eff":
        hi = float(max(np.percentile(r.n_eff, 99), 10.0))
        return r.n_eff, 1.0, hi, "Trackable points", "weighted points", True, True
    if layer == "best_matches":
        hi = float(max(np.percentile(r.best_matches, 99), 40.0))
        return r.best_matches, 1.0, hi, "Matches to best view", "matches", True, True
    if layer == "sigma_pos":
        return (r.sigma_pos * 100.0, 0.5, 50.0, "Position uncertainty",
                "centimetres", False, True)
    raise ValueError(f"unknown layer {layer!r}")


LAYERS = ("track", "expected_matched", "reloc_time", "combined", "sigma_pos",
          "best_matches", "reloc", "n_eff")
