"""Every tunable threshold in one place.

The scorers rest on constants that cannot be read out of a map file -- how
many matched points tracking needs, how much of a viewpoint change ORB
survives, when two landmarks are the same corner seen twice.  Buried in the
source they are invisible; here they can be inspected, changed, and version
controlled alongside the maps they were tuned for.

Defaults are what ``validate`` was run against, and several were measured
rather than chosen.  ``slammap2heatmap config`` prints the whole set as
annotated YAML, which is the intended way to start a project-specific file::

    ./run config > my-scans.yaml
    ./run heatmap map.msg --config my-scans.yaml
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import warnings

import yaml


class ConfigError(Exception):
    """The configuration file cannot be used as written."""


RETIRED = {
    "relocalization.pnp_sharpness": (
        "both relocalization gates are now Poisson tails, which need no width "
        "and give zero chance when nothing is expected to match"),
    "tracking.rgbd_depth_coeff": (
        "replaced by tracking.landmark_sigma_m (measured from the map by "
        "default) and tracking.sigma_floor_m"),
}
"""Settings older files may still contain, and what replaced them."""


def _f(default: Any, help: str) -> Any:
    return field(default=default, metadata={"help": help})


@dataclass
class MapConfig:
    """Deriving landmark quality from the raw map."""

    duplicate_radius_m: float = _f(0.05, (
        "Landmarks closer together than this, that also look alike, are "
        "treated as one physical feature. These maps have no loop closure, so "
        "landmark fusion never ran and each station created its own copy of "
        "the same corner; 27-40% of landmarks in the DFL scans are duplicates."))
    duplicate_hamming: int = _f(80, (
        "Descriptor distance, in bits out of 256, under which two co-located "
        "landmarks count as the same feature."))
    confusion_min_m: float = _f(0.30, (
        "Look-alikes nearer than this are ignored when scoring ambiguity: "
        "they are either the same feature or so close that matching the wrong "
        "one costs centimetres."))
    confusion_max_m: float = _f(3.0, (
        "Look-alikes further than this are ignored too: they never land in "
        "the same search window during tracking."))
    desc_neighbours: int = _f(32, (
        "How many spatial neighbours a descriptor is compared against when "
        "judging how distinctive it is."))
    viewpoint_radius_m: float = _f(0.5, (
        "Keyframes whose camera centres lie within this distance of each other "
        "count as one viewpoint. Panoramic scans place 4-8 keyframes at exactly "
        "the same spot; a hand-held or robot run drops one every few "
        "centimetres. Either way, nearby keyframes see the same landmarks and "
        "fail together, so they are pooled -- for repeatability, and for the "
        "number of independent chances relocalization gets. Scan stations in "
        "the maps tested so far are never closer than 0.69 m."))
    found_ratio_full: float = _f(0.5, (
        "Only used when the map carries real n_vis / n_fnd counters, which "
        "maps from an ordinary stella run do and scan-built maps do not. A "
        "landmark found less than this share of the times it was predicted "
        "visible is down-weighted in proportion. stella itself culls new "
        "landmarks below 0.25."))
    up_snap_deg: float = _f(2.0, (
        "If the detected gravity direction lies within this angle of a "
        "coordinate axis, snap to the axis exactly. Scan-built maps are "
        "gravity-aligned already; a hand-held run's world frame is its first "
        "camera, tilted however the camera was held, and is used as measured."))
    outlier_reproj_px: float = _f(20.0, (
        "A landmark reprojecting further than this from its own keypoint "
        "contradicts the keyframe that saw it, and is excluded. About 0.2% of "
        "landmarks per DFL map."))


@dataclass
class OccupancyConfig:
    """Recovering the floor plan."""

    scan_bin_deg: float = _f(0.75, (
        "Angular width of one synthetic scan beam. Each keyframe's landmarks "
        "are collapsed into one range per bearing before carving, which is "
        "what makes the result a floor plan rather than a starburst."))
    scan_min_hits: int = _f(3, (
        "Observations a beam needs before its range is trusted. Directions "
        "with fewer produce no evidence at all, which is what stops sparse "
        "far returns carving phantom corridors out through windows."))
    scan_percentile: float = _f(40.0, (
        "Which return in a beam is taken as the range to structure. Not the "
        "minimum, or one stray near point walls off an open direction; leans "
        "nearer than the median, because a beam through a doorway collects "
        "both the frame and the room beyond."))
    wall_band_below_m: float = _f(1.2, (
        "Metres below camera height where a landmark still counts as wall. "
        "The band is anchored to the camera, which is known exactly, rather "
        "than the floor, which has to be estimated."))
    wall_band_above_m: float = _f(0.6, (
        "Metres above camera height where a landmark still counts as wall. "
        "Keeps ceiling points from filling the plan with solid roof."))
    max_ray_m: float = _f(12.0, (
        "How far a sightline is trusted when carving free space."))
    min_free_passes: int = _f(3, (
        "How many distinct sightlines must cross a cell before it counts as "
        "open floor. One ray crossing a cell is not evidence of a room."))
    log_odds_free: float = _f(-0.45, "Evidence added to each cell a sightline passes through.")
    log_odds_occupied: float = _f(1.4, "Evidence added to the cell a sightline ends in.")
    log_odds_clamp: float = _f(8.0, "Bound on accumulated evidence, so early readings stay revisable.")
    occupied_threshold: float = _f(0.5, "Evidence above which a cell is called a wall.")


@dataclass
class VisibilityConfig:
    """Sweeping sightlines from every cell."""

    n_bearings: int = _f(120, (
        "Bearing bins in the per-cell sweep. 120 is three degrees each."))
    max_range_m: float = _f(20.0, (
        "How far a sightline is traced before it is treated as unobstructed."))


@dataclass
class TrackingConfig:
    """Staying locked on, given you already know roughly where you are."""

    view_cone_cos: float = _f(0.5, (
        "cos(60 deg): stella's limit on the angle between the viewing ray and "
        "a landmark's mean observation normal. ORB cannot match a surface "
        "seen from the far side."))
    scale_margin_lo: float = _f(0.8, "Slack below the ORB pyramid distance band.")
    scale_margin_hi: float = _f(1.2, "Slack above the ORB pyramid distance band.")
    n_tracked_marginal: float = _f(30.0, (
        "Weighted landmark count at which tracking is judged even odds. "
        "stella loses tracking below roughly 20 matched points and is "
        "comfortable above 100."))
    track_sharpness: float = _f(2.2, (
        "How sharply the tracking probability turns over around that count."))
    pixel_noise_px: float = _f(1.0, (
        "Keypoint localisation noise at pyramid level 0, in pixels."))
    landmark_sigma_m: float = _f(0.0, (
        "Position error of one map point, per axis, in metres. 0 means "
        "measure it from the map: two copies of the same corner placed "
        "independently from different viewpoints disagree by about 2.2x the "
        "per-axis error. The DFL scans measure 30-40 mm. Replaces the old "
        "rgbd_depth_coeff * d^2, which was a depth-camera model with the wrong "
        "shape for a scan-built or rendered map."))
    sigma_floor_m: float = _f(0.01, (
        "Systematic position error that no number of landmarks averages away: "
        "the map's own offset from the world, and errors shared by every "
        "point rendered from the same view. Added in quadrature. 1 cm is what "
        "the first real-run check measured (1.1-1.4 cm map-to-scan offset); "
        "`validate-runs` refits it."))
    sigma_scale: float = _f(1.0, (
        "Multiplier on the geometric part of the position uncertainty, for "
        "keypoint noise on real images exceeding pixel_noise_px. Fitted by "
        "`validate-runs`."))
    calib_a: float = _f(1.0, (
        "Calibration of the tracking probability: p -> sigmoid(a * logit(p) "
        "+ b). The identity (a=1, b=0) until fitted on real runs by "
        "`validate-runs`, which uses frames that were already tracking -- the "
        "question this layer answers -- not every frame."))
    calib_b: float = _f(0.0, "See calib_a.")
    horizon_s: float = _f(1.0, (
        "What 'keeps its lock' means: tracking survives this many seconds. "
        "`validate-runs` labels real frames the same way."))
    sigma_pos_good_m: float = _f(0.05, (
        "Position uncertainty below which geometry is not the limiting "
        "factor. This term exists to catch degenerate arrangements, such as a "
        "flat wall of points at uniform depth, not to re-rank cells that are "
        "already fine."))
    sigma_pos_max_m: float = _f(10.0, (
        "Cap, so degenerate geometry does not produce meaningless numbers."))
    bearing_bins: int = _f(360, (
        "Resolution of the internal bearing histogram, before the field-of-"
        "view convolution that turns it into per-heading counts."))


@dataclass
class RelocConfig:
    """Finding yourself from nothing."""

    bow_match_yield: float = _f(0.7, (
        "Fraction of co-visible landmarks that survive bag-of-words matching "
        "when a *real camera* frame is matched against the map. The default "
        "0.7 is the self-match value -- the map's own keyframe images against "
        "the map, which `validate` checks -- so it is an upper bound. Real "
        "frames against a rendered or scan-built map match far worse; "
        "`validate-runs` fits the real value from lost stretches in real "
        "runs, and that fitted value is the one to trust."))
    min_bow_matches: float = _f(20.0, (
        "stella's Relocalizer.min_num_bow_matches: BoW matches needed before "
        "it attempts a pose. 20 is stella's default; pass --stella-config to "
        "use the value your deployment actually runs with."))
    min_valid_obs: float = _f(50.0, (
        "stella's Relocalizer.min_num_valid_obs: landmarks that must survive "
        "pose refinement, after projection matching against the candidate "
        "keyframe has added more. A separate gate from the BoW one. 50 is "
        "stella's default; deployments often lower it."))
    find_within_s: float = _f(5.0, (
        "Time budget for 'finds itself' in the combined layer. stella "
        "attempts relocalization on every frame while lost, so a small chance "
        "per attempt still adds up: 1.4 % per frame at 30 fps is about 88 % "
        "within 5 s. Attempts are treated as independent, which is optimistic "
        "for a camera held still."))
    pooling: str = _f("max", (
        "How chances at several scan stations combine: 'max' (the best one) "
        "or 'independent' (1 - prod(1 - p)). The first real-run check showed "
        "candidates fail together -- the same frame, the same image quality -- "
        "so 'independent' inflated the probability."))
    candidates: int = _f(3, (
        "Stations considered when pooling is 'independent'."))
    viewpoint_full_deg: float = _f(30.0, (
        "Out-of-plane viewpoint change ORB tolerates without penalty. "
        "In-plane rotation costs nothing, as keypoint orientation is stored."))
    viewpoint_dead_deg: float = _f(60.0, (
        "Viewpoint change beyond which descriptor matching fails outright."))
    scale_tolerance: float = _f(1.5, (
        "Ratio of viewing distances at which matching is down-weighted by "
        "1/e. A wall seen from 2 m and from 6 m yields descriptors at "
        "different pyramid levels."))


@dataclass
class SuggestConfig:
    """Planning where to scan next."""

    reach_m: float = _f(5.0, "How far a new station is assumed to help at all.")
    decay_step_m: float = _f(0.25, (
        "Resolution of the measured effectiveness-versus-distance curve."))
    target_score: float = _f(0.5, (
        "Score a cell must reach before it stops counting as a deficit."))
    aggregate: str = _f("max", (
        "How a cell's headings collapse when planning: max, mean or min. "
        "'max' asks whether there is any direction that works, which is the "
        "right question for scan planning and the only one these maps can "
        "answer -- min-over-headings is about 0.02 everywhere."))


@dataclass
class CheckConfig:
    """What counts as passing."""

    reproj_tolerance_px: float = _f(5.0, (
        "Median reprojection error above which the pose convention is "
        "suspect and the map is refused. A wrong convention produces hundreds "
        "of pixels; a normal bundle-adjusted stella map sits at 1-4 px, and "
        "scan-built maps below 1 px."))
    depth_tolerance_rel: float = _f(0.02, (
        "Median depth disagreement, as a fraction of range, above which the "
        "stored depths are suspect. Relative, because a fixed tolerance "
        "rejects good maps that happen to be large. Skipped for monocular "
        "maps, which store no depth."))
    behind_tolerance: float = _f(0.001, (
        "Share of observations allowed to sit behind their own camera. Never "
        "zero: every DFL map carries a handful of badly placed landmarks."))
    rich_view_keypoints: int = _f(100, (
        "Landmarks a keyframe must hold before its own pose is a fair test. A "
        "scan pointed at a blank wall genuinely cannot be relocalized "
        "against, and should not be counted as a model failure."))


@dataclass
class CalibrationConfig:
    """Where the numbers came from: real runs, or model defaults."""

    matched_per_neff: float = _f(1.0, (
        "Landmarks stella actually matches per predicted trackable landmark. "
        "Turns n_eff into 'expected matched landmarks' in real units, and "
        "sets how many survive projection matching in relocalization. 1.0 "
        "until fitted; the first real-run check measured 0.89."))
    fitted: list = field(default_factory=list, metadata={"help": (
        "Which parameters were fitted on real runs. Anything not listed is a "
        "model default, and the viewer labels its layer as uncalibrated.")})
    source: str = _f("", (
        "The runs and map the fit came from, and when. Empty when nothing "
        "has been fitted."))


@dataclass
class Config:
    """Everything, in one object."""

    map: MapConfig = field(default_factory=MapConfig)
    occupancy: OccupancyConfig = field(default_factory=OccupancyConfig)
    visibility: VisibilityConfig = field(default_factory=VisibilityConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    relocalization: RelocConfig = field(default_factory=RelocConfig)
    suggest: SuggestConfig = field(default_factory=SuggestConfig)
    checks: CheckConfig = field(default_factory=CheckConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        """Read a YAML file, falling back to defaults for anything absent."""
        cfg = cls()
        if path is None:
            return cfg
        path = Path(path)
        try:
            raw = yaml.safe_load(path.read_text())
        except FileNotFoundError:
            raise ConfigError(f"{path}: no such file") from None
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: not valid YAML ({exc})") from None
        if raw is None:
            return cfg
        if not isinstance(raw, dict):
            raise ConfigError(
                f"{path}: expected sections at the top level, found "
                f"{type(raw).__name__}"
            )
        cfg._apply(raw, path)
        return cfg

    def _apply(self, raw: dict, path: Path) -> None:
        known = {f.name for f in fields(self)}
        for section, values in raw.items():
            if section not in known:
                raise ConfigError(
                    f"{path}: unknown section '{section}'. "
                    f"Expected one of: {', '.join(sorted(known))}"
                )
            target = getattr(self, section)
            if not isinstance(values, dict):
                raise ConfigError(
                    f"{path}: section '{section}' should contain settings, "
                    f"found {type(values).__name__}"
                )
            allowed = {f.name: f for f in fields(target)}
            for key, value in values.items():
                if f"{section}.{key}" in RETIRED:
                    warnings.warn(f"{path}: {section}.{key} is ignored -- "
                                  f"{RETIRED[f'{section}.{key}']}", stacklevel=3)
                    continue
                if key not in allowed:
                    raise ConfigError(
                        f"{path}: unknown setting '{section}.{key}'. "
                        f"Expected one of: {', '.join(sorted(allowed))}"
                    )
                want = allowed[key].type
                if want in ("int", int) and isinstance(value, bool):
                    raise ConfigError(f"{path}: {section}.{key} should be a number")
                if want in ("float", float, "int", int) and not isinstance(
                    value, (int, float)
                ):
                    raise ConfigError(
                        f"{path}: {section}.{key} should be a number, "
                        f"found {value!r}"
                    )
                setattr(target, key, type(getattr(target, key))(value)
                        if not isinstance(value, str) else value)
        self.validate()

    def validate(self) -> None:
        """Catch settings that are individually valid but cannot work together."""
        if self.relocalization.pooling not in ("max", "independent"):
            raise ConfigError(
                f"relocalization.pooling should be max or independent, "
                f"found {self.relocalization.pooling!r}"
            )
        if not self.calibration.matched_per_neff > 0:
            raise ConfigError("calibration.matched_per_neff must be positive")
        if self.tracking.landmark_sigma_m < 0 or self.tracking.sigma_floor_m < 0:
            raise ConfigError("tracking sigmas cannot be negative")
        if self.suggest.aggregate not in ("max", "mean", "min"):
            raise ConfigError(
                f"suggest.aggregate should be max, mean or min, "
                f"found {self.suggest.aggregate!r}"
            )
        if self.tracking.scale_margin_lo > self.tracking.scale_margin_hi:
            raise ConfigError(
                "tracking.scale_margin_lo must not exceed scale_margin_hi"
            )
        if self.map.confusion_min_m >= self.map.confusion_max_m:
            raise ConfigError(
                "map.confusion_min_m must be less than confusion_max_m"
            )
        if self.relocalization.viewpoint_full_deg >= self.relocalization.viewpoint_dead_deg:
            raise ConfigError(
                "relocalization.viewpoint_full_deg must be less than viewpoint_dead_deg"
            )
        if not 0.0 < self.tracking.view_cone_cos <= 1.0:
            raise ConfigError("tracking.view_cone_cos must be between 0 and 1")
        if self.occupancy.log_odds_free >= 0:
            raise ConfigError("occupancy.log_odds_free must be negative")
        if self.visibility.n_bearings < 8:
            raise ConfigError("visibility.n_bearings must be at least 8")

    def apply_stella(self, path: str | Path) -> list[str]:
        """Take thresholds from the stella_vslam config the device actually runs.

        The heatmap models stella's relocalizer, so its gates should be the
        deployed ones, not stella's defaults. Returns what was applied.
        """
        path = Path(path)
        try:
            text = path.read_text()
        except FileNotFoundError:
            raise ConfigError(f"{path}: no such file") from None
        # OpenCV-style YAML headers trip PyYAML; stella's own files rarely have one.
        text = "\n".join(ln for ln in text.splitlines() if not ln.startswith("%YAML"))
        try:
            doc = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: not valid YAML ({exc})") from None
        reloc = doc.get("Relocalizer") or {}
        applied = []
        for stella_key, ours in (("min_num_bow_matches", "min_bow_matches"),
                                 ("min_num_valid_obs", "min_valid_obs")):
            if stella_key in reloc:
                setattr(self.relocalization, ours, float(reloc[stella_key]))
                applied.append(f"relocalization.{ours} = {reloc[stella_key]} "
                               f"(Relocalizer.{stella_key})")
        self.validate()
        return applied

    @staticmethod
    def stella_camera(path: str | Path) -> dict | None:
        """The Camera block of a stella config, if it has one."""
        text = "\n".join(ln for ln in Path(path).read_text().splitlines()
                         if not ln.startswith("%YAML"))
        cam = (yaml.safe_load(text) or {}).get("Camera")
        return cam if isinstance(cam, dict) else None

    def to_dict(self) -> dict:
        return {
            f.name: {g.name: getattr(getattr(self, f.name), g.name)
                     for g in fields(getattr(self, f.name))}
            for f in fields(self)
        }

    def to_yaml(self, comments: bool = True) -> str:
        """Render as YAML, optionally with every setting's rationale inline."""
        out: list[str] = []
        if comments:
            out += [
                "# slammap2heatmap configuration.",
                "# Every value below is the default; delete any you do not want to change.",
                "#   ./run heatmap map.msg --config this-file.yaml",
                "",
            ]
        for sec in fields(self):
            target = getattr(self, sec.name)
            if comments and is_dataclass(target):
                doc = (target.__doc__ or "").strip().splitlines()[0]
                out.append(f"# {doc}")
            out.append(f"{sec.name}:")
            for g in fields(target):
                if comments:
                    text = g.metadata.get("help", "")
                    for line in _wrap(text, 74):
                        out.append(f"  # {line}")
                out.append(f"  {g.name}: {_scalar(getattr(target, g.name))}")
                if comments:
                    out.append("")
            if not comments:
                out.append("")
        return "\n".join(out).rstrip() + "\n"


def _scalar(value: Any) -> str:
    """One value as YAML. ``safe_dump`` would append a document-end marker."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value if value.isidentifier() else repr(value)
    return repr(value)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > width:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    if cur:
        lines.append(cur)
    return lines


DEFAULT = Config()
"""The settings everything falls back to, and what ``validate`` was run against."""
