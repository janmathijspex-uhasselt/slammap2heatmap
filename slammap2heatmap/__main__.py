"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import (export, grid as gridmod, loader, mapmodel, occupancy,
               render, runs, scorers, selfcheck, suggest, validate,
               visibility)
from .config import Config, ConfigError
from .loader import MapFormatError


def _maps(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        out.extend(sorted(path.glob("*.msg")) if path.is_dir() else [path])
    if not out:
        where = ", ".join(paths)
        raise SystemExit(f"no .msg files found in {where}")
    return out


def _config(args: argparse.Namespace) -> Config:
    return Config.load(getattr(args, "config", None))


def _load(path: Path, args: argparse.Namespace):
    return loader.load(path, scale=getattr(args, "scale", None))


def _require_metric(raw, path: Path) -> bool:
    """Refuse maps the scorers would quietly get wrong.

    Two cases. Everything spatial is in metres, so a monocular map with no
    scale would produce confident nonsense. And the virtual camera flown
    through each cell *is the map's own camera*: for a 360-degree map that
    means a device that sees in every direction at once, which no phone does.
    """
    if raw.camera.is_spherical:
        print(f"{path.name}: built with a 360-degree camera. Scoring simulates a "
              f"device with the map's own camera, which here would see every "
              f"heading at once and rate everything as excellent. `inspect` "
              f"works; scoring 360-degree maps is not supported yet.",
              file=sys.stderr)
        return False
    if raw.is_metric:
        return True
    print(f"{path.name}: monocular map with no scale - its lengths are in "
          f"arbitrary units, so grid size, wall heights, viewing distances and "
          f"station reach would all be wrong.\n"
          f"  Measure one known distance in the map and pass "
          f"--scale <true metres / map units>,\n"
          f"  or --scale 1 if you know it is already metric.", file=sys.stderr)
    return False


def cmd_config(args: argparse.Namespace) -> int:
    """Print the settings, annotated, as a starting point for a project file."""
    print(Config.load(args.config).to_yaml(comments=not args.bare), end="")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    cfg = _config(args)
    failed: list[str] = []
    for path in _maps(args.map):
        try:
            raw = _load(path, args)
        except MapFormatError as exc:
            print(f"{path.name}\n  skipped: {exc}\n", file=sys.stderr)
            failed.append(path.name)
            continue
        model = mapmodel.build(raw, up_axis=args.up, cfg=cfg)
        check = selfcheck.check_geometry(raw, cfg=cfg)
        print(selfcheck.summarise(model, check))
        print()
        if not check.ok:
            failed.append(path.name)
    if failed:
        print(f"{len(failed)} map(s) could not be used: {', '.join(failed)}",
              file=sys.stderr)
    return 1 if failed else 0


def cmd_heatmap(args: argparse.Namespace) -> int:
    import numpy as np

    cfg = _config(args)
    out_root = Path(args.out)
    written: list[Path] = []
    for path in _maps(args.map):
        try:
            raw = _load(path, args)
        except MapFormatError as exc:
            print(f"{path.name}: skipped, {exc}", file=sys.stderr)
            continue

        if not _require_metric(raw, path):
            continue
        _apply_stella(cfg, args, raw)
        model = mapmodel.build(raw, up_axis=args.up, cfg=cfg)
        check = selfcheck.check_geometry(raw, cfg=cfg)
        if not check.ok:
            print(f"{path.name}: {check.describe()}", file=sys.stderr)
            print("  refusing to build a heatmap from a map that fails its own "
                  "geometry check", file=sys.stderr)
            continue

        height = args.height if args.height is not None else model.station_h
        print(f"{path.name}")
        grid = gridmod.fit(model, res=args.res)
        print(f"  grid        {grid.nx} x {grid.ny} cells at {grid.res * 100:.0f} cm")

        occ = occupancy.build(model, grid, cfg=cfg)
        print(f"  floor plan  {occ.summary()}")

        cells = np.flatnonzero(occ.free.ravel())
        if cells.size == 0:
            print("  no open floor was recovered; nothing to score",
                  file=sys.stderr)
            continue

        vis = visibility.sweep(occ, cells, cfg=cfg)
        print(f"  visibility  {vis.summary()}")

        result = scorers.evaluate(
            model, grid, occ, vis, height=height, yaw_bins=args.yaw_bins,
            use_occlusion=not args.no_occlusion, combine=args.combine, cfg=cfg,
        )
        print(f"  scored      {result.summary()}")

        if args.png == []:
            args.png = ["track"]

        plan = None
        if args.suggest:
            plan = suggest.recommend(model, grid, vis, result, n=args.suggest,
                                     aggregate=args.aggregate, cfg=cfg)
            print(plan.describe())

        if args.validate:
            report = validate.run(model, grid, occ, vis, result, height, cfg=cfg)
            print(report.describe())

        paths = export.write(
            out_root, model, occ, result, plan=plan, cfg=cfg,
            params={
                "res": args.res, "yaw_bins": args.yaw_bins, "height": height,
                "occlusion": not args.no_occlusion, "up": model.up_axis,
                "combine": args.combine,
                "config": cfg.to_dict(),
            },
        )
        if args.png:
            layers = (render.LAYERS if args.png == ["all"] else args.png)
            for name in layers:
                png = render.layer_png(
                    out_root / f"{path.stem}_{name}.png", model, occ, result,
                    layer=name, aggregate=args.png_aggregate,
                    scale=args.png_scale, plan=plan,
                    params={"combine": args.combine,
                            "occlusion": not args.no_occlusion,
                            "yaw_bins": args.yaw_bins},
                )
                print(f"  png         {png}")
        print(f"  viewer      {paths['html']}\n")
        written.append(paths["html"])

    if not written:
        return 1
    if args.open:
        import webbrowser
        webbrowser.open(written[0].resolve().as_uri())
    else:
        print("open it with:")
        for p in written:
            print(f"  xdg-open {p}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    import numpy as np

    cfg = _config(args)
    failed = 0
    for path in _maps(args.map):
        try:
            raw = _load(path, args)
        except MapFormatError as exc:
            print(f"{path.name}: skipped, {exc}", file=sys.stderr)
            failed += 1
            continue

        if not _require_metric(raw, path):
            failed += 1
            continue
        _apply_stella(cfg, args, raw)
        model = mapmodel.build(raw, up_axis=args.up, cfg=cfg)
        check = selfcheck.check_geometry(raw, cfg=cfg)
        height = args.height if args.height is not None else model.station_h
        gridd = gridmod.fit(model, res=args.res)
        occ = occupancy.build(model, gridd, cfg=cfg)
        cells = np.flatnonzero(occ.free.ravel())
        print(f"{path.name}")
        print(f"  {check.describe()}")
        if cells.size == 0:
            print("  no open floor was recovered; nothing to validate\n",
                  file=sys.stderr)
            failed += 1
            continue
        vis = visibility.sweep(occ, cells, cfg=cfg)
        result = scorers.evaluate(
            model, gridd, occ, vis, height=height, yaw_bins=args.yaw_bins, cfg=cfg,
        )
        report = validate.run(model, gridd, occ, vis, result, height,
                              holdout=not args.no_holdout, cfg=cfg)
        print(report.describe())
        print()
        if not (check.ok and report.poses.ok):
            failed += 1

    if failed:
        print(f"{failed} of the maps did not pass. A map can fail while the "
              f"others pass only if that map is weak: the thresholds are "
              f"shared, so a wrong model would fail all of them.",
              file=sys.stderr)
    return 1 if failed else 0


def cmd_suggest(args: argparse.Namespace) -> int:
    import numpy as np

    cfg = _config(args)
    refused = 0
    for path in _maps(args.map):
        try:
            raw = _load(path, args)
        except MapFormatError as exc:
            print(f"{path.name}: skipped, {exc}", file=sys.stderr)
            continue
        if not _require_metric(raw, path):
            refused += 1
            continue
        _apply_stella(cfg, args, raw)
        model = mapmodel.build(raw, up_axis=args.up, cfg=cfg)
        height = args.height if args.height is not None else model.station_h
        gridd = gridmod.fit(model, res=args.res)
        occ = occupancy.build(model, gridd, cfg=cfg)
        cells = np.flatnonzero(occ.free.ravel())
        print(f"{path.name}")
        if cells.size == 0:
            print("  no open floor was recovered\n", file=sys.stderr)
            continue
        vis = visibility.sweep(occ, cells, cfg=cfg)
        result = scorers.evaluate(model, gridd, occ, vis, height=height,
                                  yaw_bins=args.yaw_bins, cfg=cfg)
        plan = suggest.recommend(model, gridd, vis, result, n=args.n, cfg=cfg,
                                 reach=args.reach, target=args.target,
                                 aggregate=args.aggregate)
        print(plan.describe())
        print()
    return 1 if refused else 0


def _apply_stella(cfg, args, raw=None) -> None:
    """Take relocalizer thresholds from the stella config actually deployed."""
    path = getattr(args, "stella_config", None)
    if not path:
        return
    for line in cfg.apply_stella(path):
        print(f"  from {Path(path).name}: {line}")
    cam = cfg.stella_camera(path)
    if raw is not None and cam and "fx" in cam and "cols" in cam:
        import numpy as np
        dev = float(np.degrees(2 * np.arctan(0.5 * float(cam["cols"]) / float(cam["fx"]))))
        mapped = raw.camera.hfov_deg
        if abs(dev - mapped) / mapped > 0.05:
            print(f"  WARNING: the device camera sees {dev:.1f} deg across, the map's "
                  f"{mapped:.1f} deg. The heatmap simulates the map's camera, so "
                  f"expect it to be off by roughly that ratio.", file=sys.stderr)


def cmd_validate_runs(args: argparse.Namespace) -> int:
    import numpy as np

    cfg = _config(args)
    path = Path(args.map)
    try:
        raw = _load(path, args)
    except MapFormatError as exc:
        print(f"{path.name}: {exc}", file=sys.stderr)
        return 1
    if not _require_metric(raw, path):
        return 1
    _apply_stella(cfg, args, raw)

    model = mapmodel.build(raw, up_axis=args.up, cfg=cfg)
    height = args.height if args.height is not None else model.station_h
    grid = gridmod.fit(model, res=args.res)
    occ = occupancy.build(model, grid, cfg=cfg)
    cells = np.flatnonzero(occ.free.ravel())
    if cells.size == 0:
        print("no open floor was recovered; nothing to compare", file=sys.stderr)
        return 1
    vis = visibility.sweep(occ, cells, cfg=cfg)

    try:
        fr = runs.read_frames(args.frames)
    except (OSError, ValueError) as exc:
        print(f"cannot read frames: {exc}", file=sys.stderr)
        return 1
    run_names = sorted(set(fr.run.tolist()))
    horizon = args.horizon if args.horizon is not None else cfg.tracking.horizon_s

    # Only the cells the runs visited need scoring.
    flat = grid.sample_flat(np.nan_to_num(fr.xy, nan=1e9))
    touched = np.isin(vis.cells, flat[flat >= 0])
    vis_runs = vis.subset(touched)

    def score(c):
        r = scorers.evaluate(model, grid, occ, vis_runs, height=height,
                             yaw_bins=args.yaw_bins, cfg=c)
        at = runs.lookup(r, fr)
        return r, at, runs.check(fr, at, horizon, n_boot=args.bootstrap,
                                 reference_cells=vis.cells)

    result, at, rep = score(cfg)
    print(runs.describe_header(rep, model, run_names))
    calibrated = ", ".join(cfg.calibration.fitted) or "nothing"
    print(runs.describe(rep, f"as configured (calibrated on real runs: {calibrated})"))

    out_cfg = cfg
    if args.fit:
        f = runs.fit(model, grid, occ, vis_runs, result, fr, at, cfg,
                     height, horizon, n_boot=args.bootstrap)
        print(runs.describe_fit(f, cfg))
        out_cfg = f.apply(cfg)
        out_cfg.calibration.source = runs.provenance(model, run_names)
        _, _, rep2 = score(out_cfg)
        print(runs.describe(rep2, "after fitting -- measured on the same frames it "
                                  "was fitted on, so optimistic; check on a new run"))

    if args.write_config:
        Path(args.write_config).write_text(out_cfg.to_yaml())
        print(f"  wrote {args.write_config} -- use it with "
              f"`heatmap --config {args.write_config}`")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="slammap2heatmap",
        description="Localizability heatmaps from stella_vslam map databases.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "inspect",
        help="decode a map, verify its geometry, and report what it contains",
    )
    p.add_argument("map", nargs="+", help=".msg file(s) or a directory of them")
    p.add_argument("--up", choices=["x", "y", "z", "-x", "-y", "-z"], default=None,
                   help="world up axis (default: inferred from camera orientations)")
    p.set_defaults(func=cmd_inspect)

    h = sub.add_parser(
        "heatmap",
        help="build the floor plan and localizability heatmap, and a viewer",
    )
    h.add_argument("map", nargs="+", help=".msg file(s) or a directory of them")
    h.add_argument("-o", "--out", default="out", help="output directory (default: out)")
    h.add_argument("--res", type=float, default=0.15,
                   help="grid cell size in metres (default: 0.15)")
    h.add_argument("--yaw-bins", type=int, default=16,
                   help="headings scored per cell (default: 16)")
    h.add_argument("--height", type=float, default=None,
                   help="camera height to simulate (default: the scanner's own)")
    h.add_argument("--combine", choices=["product", "min", "sum"], default="product",
                   help="how to merge relocalization and tracking "
                        "(default: product - both must work)")
    h.add_argument("--suggest", type=int, nargs="?", const=5, default=0,
                   metavar="N",
                   help="also recommend N places to put the next scan station "
                        "(default 5 when given without a number)")
    h.add_argument("--aggregate", choices=["max", "mean", "min"], default=None,
                   help="how headings collapse when planning scans "
                        "(default: max - is there any direction that works)")
    h.add_argument("--validate", action="store_true",
                   help="also run the validation checks")
    h.add_argument("--no-occlusion", action="store_true",
                   help="ignore walls when deciding what is visible")
    h.add_argument("--up", choices=["x", "y", "z", "-x", "-y", "-z"], default=None,
                   help="world up axis (default: inferred)")
    h.add_argument("--png", nargs="*", default=None,
                   metavar="LAYER", choices=[*render.LAYERS, "all"],
                   help="also write report-ready PNGs with scale bar, legend "
                        "and settings burned in. Give layer names, or nothing "
                        f"for track. Available: {', '.join(render.LAYERS)}")
    h.add_argument("--png-aggregate", choices=["max", "mean", "min"],
                   default="mean",
                   help="how headings collapse in the PNG: max = best heading, "
                        "min = worst, whichever way the layer points "
                        "(default: mean)")
    h.add_argument("--png-scale", type=int, default=4,
                   help="pixels per grid cell in the PNG (default: 4)")
    h.add_argument("--open", action="store_true",
                   help="open the viewer in a browser when done")
    h.set_defaults(func=cmd_heatmap)

    v = sub.add_parser(
        "validate",
        help="check the predictions against the map's own ground truth",
    )
    v.add_argument("map", nargs="+", help=".msg file(s) or a directory of them")
    v.add_argument("--res", type=float, default=0.15,
                   help="grid cell size in metres (default: 0.15)")
    v.add_argument("--yaw-bins", type=int, default=16,
                   help="headings scored per cell (default: 16)")
    v.add_argument("--height", type=float, default=None,
                   help="camera height to simulate (default: the scanner's own)")
    v.add_argument("--no-holdout", action="store_true",
                   help="skip leave-one-station-out, which is the slow part")
    v.add_argument("--up", choices=["x", "y", "z", "-x", "-y", "-z"], default=None,
                   help="world up axis (default: inferred)")
    v.set_defaults(func=cmd_validate)

    g = sub.add_parser(
        "suggest",
        help="recommend where to put the next scan positions",
    )
    g.add_argument("map", nargs="+", help=".msg file(s) or a directory of them")
    g.add_argument("-n", type=int, default=5, help="how many to suggest (default: 5)")
    g.add_argument("--res", type=float, default=0.15, help="grid cell size in metres")
    g.add_argument("--yaw-bins", type=int, default=16, help="headings scored per cell")
    g.add_argument("--height", type=float, default=None, help="camera height to simulate")
    g.add_argument("--reach", type=float, default=None,
                   help="how far a new station is assumed to help (metres)")
    g.add_argument("--target", type=float, default=None,
                   help="score a cell must reach to stop counting as weak")
    g.add_argument("--aggregate", choices=["max", "mean", "min"], default=None,
                   help="how headings collapse when planning (default: max)")
    g.add_argument("--up", choices=["x", "y", "z", "-x", "-y", "-z"], default=None,
                   help="world up axis (default: inferred)")
    g.set_defaults(func=cmd_suggest)

    vr = sub.add_parser(
        "validate-runs",
        help="compare the heatmap with real localization runs, and calibrate on them",
    )
    vr.add_argument("map", help="the .msg map the runs localized against")
    vr.add_argument("--frames", nargs="+", required=True, metavar="CSV",
                    help="frames.csv from the alignment app (one row per video frame)")
    vr.add_argument("--fit", action="store_true",
                    help="fit the parameters real images decide, and show the result")
    vr.add_argument("--write-config", metavar="FILE",
                    help="write the (fitted) settings to this YAML file")
    vr.add_argument("--horizon", type=float, default=None,
                    help="seconds a lock must hold to count as kept "
                         "(default: tracking.horizon_s)")
    vr.add_argument("--res", type=float, default=0.15, help="grid cell size in metres")
    vr.add_argument("--yaw-bins", type=int, default=16, help="headings per cell")
    vr.add_argument("--height", type=float, default=None, help="camera height to simulate")
    vr.add_argument("--bootstrap", type=int, default=runs.N_BOOT,
                    help="bootstrap resamples for the intervals")
    vr.add_argument("--up", choices=["x", "y", "z", "-x", "-y", "-z"], default=None)
    vr.set_defaults(func=cmd_validate_runs)

    cf = sub.add_parser(
        "config",
        help="print every tunable setting, annotated, as a starting point",
    )
    cf.add_argument("--config", default=None,
                    help="an existing file to fill in from, rather than defaults")
    cf.add_argument("--bare", action="store_true",
                    help="omit the explanatory comments")
    cf.set_defaults(func=cmd_config)

    for parser_with_config in (p, h, v, g, vr):
        parser_with_config.add_argument(
            "--config", default=None, metavar="FILE",
            help="YAML settings file; see `slammap2heatmap config`",
        )
        parser_with_config.add_argument(
            "--stella-config", default=None, metavar="YAML",
            help="the stella_vslam config the device runs; relocalizer "
                 "thresholds are taken from it",
        )
        parser_with_config.add_argument(
            "--scale", type=float, default=None, metavar="FACTOR",
            help="multiply every length in the map by this; required for "
                 "monocular maps, whose units are arbitrary",
        )

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
