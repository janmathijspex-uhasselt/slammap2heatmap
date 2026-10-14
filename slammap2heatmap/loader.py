"""Read a stella_vslam map database (.msg) into typed arrays.

Everything the format gets wrong-footed by is handled here, so that no other
module has to know about it:

* ``keyframes`` / ``landmarks`` are keyed by *strings* while ``lm_ids`` holds
  *integers* -- comparing the two directly silently matches nothing.
* ``lm_ids`` is as long as the *original* ORB keypoint count.  The surviving
  ids are packed to the front and the tail is padded with -1, so only the
  first ``n_keypts`` entries line up with ``undist_keypts`` / ``descs`` /
  ``depths``.
* ``span_parent`` uses 4294967295 (uint32 -1) for "no parent".
* ``lm_ids`` may reference landmarks that are not in the file.

Pose convention, verified by reprojection on all six DFL maps (median error
0.26 px): ``rot_cw`` is a quaternion in (x, y, z, w) order and ``trans_cw``
the matching translation, both world -> camera.  The camera centre in world
coordinates is ``-R.T @ t``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import msgpack
import numpy as np

class MapFormatError(Exception):
    """The file is not a map database this tool can read."""


NO_PARENT = 4294967295
"""uint32 sentinel stella writes into ``span_parent`` when there is none."""


@dataclass(frozen=True)
class Camera:
    """Intrinsics of one camera model in the map."""

    name: str
    model_type: str
    setup_type: str
    cols: int
    rows: int
    fx: float
    fy: float
    cx: float
    cy: float
    fps: float

    @property
    def is_spherical(self) -> bool:
        return self.model_type == "Equirectangular"

    @property
    def focal_px(self) -> float:
        """Pixels per radian at the image centre.

        For a pinhole this is the focal length.  An equirectangular image has
        none, but spreads ``cols`` pixels over a full turn, which is what the
        uncertainty model actually needs.
        """
        if self.is_spherical:
            return self.cols / (2.0 * np.pi)
        return 0.5 * (self.fx + self.fy)

    def project(self, pc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Camera-frame points -> pixels, plus which ones are in view.

        Fisheye maps store keypoints already undistorted onto the pinhole
        plane of the same ``K``, so they project like a perspective camera.
        """
        pc = np.asarray(pc, dtype=np.float64)
        if self.is_spherical:
            r = np.maximum(np.linalg.norm(pc, axis=1), 1e-12)
            u = self.cols * (0.5 + np.arctan2(pc[:, 0], pc[:, 2]) / (2 * np.pi))
            v = self.rows * (0.5 + np.arcsin(np.clip(pc[:, 1] / r, -1, 1)) / np.pi)
            return np.column_stack([u, v]), np.ones(len(pc), dtype=bool)
        z = pc[:, 2]
        front = z > 0
        zz = np.where(front, z, 1.0)
        uv = np.column_stack([self.fx * pc[:, 0] / zz + self.cx,
                              self.fy * pc[:, 1] / zz + self.cy])
        return uv, front

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    @property
    def hfov_deg(self) -> float:
        if self.is_spherical:
            return 360.0
        if self.model_type == "Fisheye":
            # Equidistant approximation, r = f * theta; distortion ignored.
            return float(min(np.degrees(self.cols / self.fx), 359.0))
        return float(np.degrees(2.0 * np.arctan(0.5 * self.cols / self.fx)))

    @property
    def vfov_deg(self) -> float:
        if self.is_spherical:
            return 180.0
        if self.model_type == "Fisheye":
            return float(min(np.degrees(self.rows / self.fy), 179.0))
        return float(np.degrees(2.0 * np.arctan(0.5 * self.rows / self.fy)))


@dataclass(frozen=True)
class OrbParams:
    """ORB pyramid description, needed for the per-octave scale band."""

    name: str
    scale_factor: float
    num_levels: int
    ini_fast_threshold: int
    min_fast_threshold: int

    @property
    def scale_factors(self) -> np.ndarray:
        """``scale_factor ** level`` for every pyramid level."""
        return self.scale_factor ** np.arange(self.num_levels, dtype=np.float64)


@dataclass
class RawMap:
    """A map file, decoded but not yet interpreted.

    Keyframes and landmarks are stored column-wise.  ``kf_ids`` / ``lm_ids``
    hold the original stella ids; everything else is indexed by *row*, and
    :attr:`kf_index` / :attr:`lm_index` translate id -> row.  The observation
    table at the bottom is the flattened (keyframe, keypoint) -> landmark
    relation with dangling references already dropped.
    """

    path: Path
    cameras: dict[str, Camera]
    orb_params: dict[str, OrbParams]

    # --- keyframes, one row each ------------------------------------------
    kf_ids: np.ndarray            # (K,) int64, original stella ids
    kf_index: dict[int, int]      # stella id -> row
    kf_quat_cw: np.ndarray        # (K, 4) float64, (x, y, z, w)
    kf_trans_cw: np.ndarray       # (K, 3) float64
    kf_rot_cw: np.ndarray         # (K, 3, 3) float64
    kf_center: np.ndarray         # (K, 3) float64, world coordinates
    kf_n_keypts: np.ndarray       # (K,) int64, surviving keypoints
    kf_n_detected: np.ndarray     # (K,) int64, len(lm_ids) = ORB keypoints found
    kf_ts: np.ndarray             # (K,) float64
    kf_cam: list[str]             # (K,) camera name per keyframe
    kf_orb: list[str]             # (K,) orb_params name per keyframe
    kf_span_parent: np.ndarray    # (K,) int64, -1 where absent
    kf_loop_edges: list[list[int]]

    # --- landmarks, one row each ------------------------------------------
    lm_ids: np.ndarray            # (L,) int64
    lm_index: dict[int, int]      # stella id -> row
    lm_pos_w: np.ndarray          # (L, 3) float64
    lm_n_vis: np.ndarray          # (L,) int64  -- all 1 in the DFL maps
    lm_n_fnd: np.ndarray          # (L,) int64  -- all 1 in the DFL maps
    lm_ref_kf: np.ndarray         # (L,) int64, stella keyframe id
    lm_first_kf: np.ndarray       # (L,) int64, stella keyframe id

    # --- observations, one row per (keyframe, keypoint) pair ---------------
    obs_kf: np.ndarray            # (N,) int64, keyframe *row*
    obs_lm: np.ndarray            # (N,) int64, landmark *row*
    obs_pt: np.ndarray            # (N, 2) float64, undistorted pixel
    obs_octave: np.ndarray        # (N,) int8, ORB pyramid level
    obs_angle: np.ndarray         # (N,) float32, degrees
    obs_depth: np.ndarray         # (N,) float32, metric depth (RGBD)
    obs_desc: np.ndarray          # (N, 32) uint8, 256-bit ORB descriptor

    n_dangling: int               # lm_ids entries pointing at absent landmarks
    scale: float = 1.0            # factor applied at load, map units -> metres
    scale_given: bool = False     # whether someone vouched for the units

    # ------------------------------------------------------------------
    @property
    def n_keyframes(self) -> int:
        return len(self.kf_ids)

    @property
    def n_landmarks(self) -> int:
        return len(self.lm_ids)

    @property
    def n_observations(self) -> int:
        return len(self.obs_kf)

    @property
    def is_metric(self) -> bool:
        """Whether one map unit is one metre.

        A monocular map is reconstructed only up to scale: stella sets the
        initial median depth to 1 and everything follows from that.  RGBD and
        stereo maps take their scale from the sensor.  A monocular map that
        was given an explicit ``scale`` at load counts as metric.
        """
        return self.camera.setup_type != "Monocular" or self.scale_given

    @property
    def has_depth(self) -> bool:
        return bool(np.isfinite(self.obs_depth).any())

    @property
    def camera(self) -> Camera:
        """The single camera, for the common case of a one-camera map."""
        if len(self.cameras) != 1:
            raise ValueError(
                f"map has {len(self.cameras)} cameras; use .cameras[name]"
            )
        return next(iter(self.cameras.values()))

    @property
    def orb(self) -> OrbParams:
        if len(self.orb_params) != 1:
            raise ValueError(
                f"map has {len(self.orb_params)} orb_params; use .orb_params[name]"
            )
        return next(iter(self.orb_params.values()))


def quats_to_rotations(q: np.ndarray) -> np.ndarray:
    """Convert (N, 4) quaternions in (x, y, z, w) order to (N, 3, 3) matrices."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _descs_to_bytes(descs: list) -> np.ndarray:
    """(n, 8) uint32 words as stella writes them -> (n, 32) uint8."""
    if not descs:
        return np.zeros((0, 32), dtype=np.uint8)
    words = np.asarray(descs, dtype=np.uint32)
    return words.view(np.uint8).reshape(len(words), -1)[:, :32]


SUPPORTED_MODELS = ("Perspective", "Fisheye", "Equirectangular")


def _camera(name: str, c: dict, path: Path) -> Camera:
    model = c.get("model_type", "?")
    if model not in SUPPORTED_MODELS:
        raise MapFormatError(
            f"{path.name}: camera model '{model}' is not supported "
            f"(supported: {', '.join(SUPPORTED_MODELS)})"
        )
    if model == "Equirectangular":
        # No intrinsics to speak of: the image is a full sphere.
        cols, rows = int(c["cols"]), int(c["rows"])
        fx = fy = cols / (2.0 * np.pi)
        cx, cy = cols / 2.0, rows / 2.0
    else:
        fx, fy, cx, cy = (float(c[k]) for k in ("fx", "fy", "cx", "cy"))
    return Camera(
        name=name, model_type=model, setup_type=c.get("setup_type", "Monocular"),
        cols=int(c["cols"]), rows=int(c["rows"]),
        fx=fx, fy=fy, cx=cx, cy=cy, fps=float(c.get("fps", 0.0)),
    )


def load(path: str | Path, scale: float | None = None) -> RawMap:
    """Decode a stella_vslam ``.msg`` map database.

    ``scale`` multiplies every length in the map -- positions, translations
    and depths -- and exists for monocular maps, whose units are arbitrary.
    Measure one known distance in the map, divide the true distance by it,
    and pass that.  Passing ``1.0`` declares a monocular map already metric.
    """
    scale_given = scale is not None
    scale = 1.0 if scale is None else float(scale)
    if not scale > 0:
        raise MapFormatError(f"scale must be positive, got {scale}")
    path = Path(path)
    if not path.exists():
        raise MapFormatError(f"{path}: no such file")
    if path.is_dir():
        raise MapFormatError(f"{path}: is a directory, not a .msg file")

    try:
        with open(path, "rb") as fh:
            blob = msgpack.unpackb(fh.read(), raw=False, strict_map_key=False)
    except OSError as exc:
        raise MapFormatError(f"{path}: cannot be read ({exc.strerror})") from exc
    except Exception as exc:                      # msgpack raises several types
        raise MapFormatError(
            f"{path.name}: not valid msgpack, so not a stella_vslam map "
            f"({type(exc).__name__})"
        ) from exc

    if not isinstance(blob, dict):
        raise MapFormatError(
            f"{path.name}: expected a msgpack map at the top level, "
            f"found {type(blob).__name__}"
        )
    missing = {"cameras", "keyframes", "landmarks", "orb_params"} - set(blob)
    if missing:
        raise MapFormatError(
            f"{path.name}: not a stella_vslam map database, missing "
            f"{', '.join(sorted(missing))}"
        )

    cameras = {str(name): _camera(str(name), c, path)
               for name, c in blob["cameras"].items()}
    orb_params = {
        str(name): OrbParams(
            name=str(name),
            scale_factor=float(o["scale_factor"]),
            num_levels=int(o["num_levels"]),
            ini_fast_threshold=int(o.get("ini_fast_threshold", 20)),
            min_fast_threshold=int(o.get("min_fast_threshold", 7)),
        )
        for name, o in blob["orb_params"].items()
    }

    # --- landmarks ---------------------------------------------------------
    lm_raw = blob["landmarks"]
    lm_ids = np.fromiter((int(k) for k in lm_raw), dtype=np.int64, count=len(lm_raw))
    order = np.argsort(lm_ids, kind="stable")
    lm_ids = lm_ids[order]
    lm_index = {int(v): i for i, v in enumerate(lm_ids)}
    lm_vals = list(lm_raw.values())
    lm_vals = [lm_vals[i] for i in order]

    lm_pos_w = np.asarray([v["pos_w"] for v in lm_vals], dtype=np.float64) * scale
    lm_n_vis = np.asarray([v.get("n_vis", 0) for v in lm_vals], dtype=np.int64)
    lm_n_fnd = np.asarray([v.get("n_fnd", 0) for v in lm_vals], dtype=np.int64)
    lm_ref_kf = np.asarray([v.get("ref_keyfrm", -1) for v in lm_vals], dtype=np.int64)
    lm_first_kf = np.asarray([v.get("1st_keyfrm", -1) for v in lm_vals], dtype=np.int64)

    # --- keyframes ---------------------------------------------------------
    kf_raw = blob["keyframes"]
    kf_ids = np.fromiter((int(k) for k in kf_raw), dtype=np.int64, count=len(kf_raw))
    order = np.argsort(kf_ids, kind="stable")
    kf_ids = kf_ids[order]
    kf_index = {int(v): i for i, v in enumerate(kf_ids)}
    kf_vals = list(kf_raw.values())
    kf_vals = [kf_vals[i] for i in order]

    kf_quat_cw = np.asarray([v["rot_cw"] for v in kf_vals], dtype=np.float64)
    kf_trans_cw = np.asarray([v["trans_cw"] for v in kf_vals], dtype=np.float64) * scale
    kf_rot_cw = quats_to_rotations(kf_quat_cw)
    # centre = -R^T t, batched
    kf_center = -np.einsum("kji,kj->ki", kf_rot_cw, kf_trans_cw)

    kf_n_keypts = np.asarray([int(v["n_keypts"]) for v in kf_vals], dtype=np.int64)
    kf_n_detected = np.asarray([len(v["lm_ids"]) for v in kf_vals], dtype=np.int64)
    kf_ts = np.asarray([float(v.get("ts", 0.0)) for v in kf_vals], dtype=np.float64)
    kf_cam = [str(v["cam"]) for v in kf_vals]
    kf_orb = [str(v.get("orb_params", "default")) for v in kf_vals]
    kf_span_parent = np.asarray(
        [-1 if v.get("span_parent", NO_PARENT) == NO_PARENT else int(v["span_parent"])
         for v in kf_vals],
        dtype=np.int64,
    )
    kf_loop_edges = [[int(e) for e in v.get("loop_edges", [])] for v in kf_vals]

    # --- observations ------------------------------------------------------
    obs_kf_l, obs_lm_l, obs_pt_l = [], [], []
    obs_oct_l, obs_ang_l, obs_dep_l, obs_desc_l = [], [], [], []
    n_dangling = 0

    for row, v in enumerate(kf_vals):
        n = int(v["n_keypts"])
        if n == 0:
            continue
        # Stock stella writes one lm_id per keypoint with -1 where nothing
        # matched; CRISTAL-built maps pack the matched ones to the front and
        # pad to the original detection count.  Taking the first n_keypts
        # entries is correct for both.
        ids = v["lm_ids"][:n]
        kpts = v["undist_keypts"]
        depths = v.get("depths") or []
        if not (len(kpts) == len(v["descs"]) == n):
            raise MapFormatError(
                f"{path.name}: keyframe {kf_ids[row]} is inconsistent: "
                f"n_keypts={n} but {len(kpts)} keypoints, "
                f"{len(v['descs'])} descriptors"
            )
        if depths and len(depths) != n:
            raise MapFormatError(
                f"{path.name}: keyframe {kf_ids[row]} has {len(depths)} depths "
                f"for {n} keypoints"
            )
        # Monocular frames store no depths at all; RGBD and stereo store -1
        # where the sensor returned nothing.  Both become NaN.
        depth = (np.asarray(depths, dtype=np.float32) if depths
                 else np.full(n, np.nan, dtype=np.float32))
        depth[~(depth > 0)] = np.nan

        ids_arr = np.asarray(ids, dtype=np.int64)
        rows = np.fromiter(
            (lm_index.get(int(i), -1) for i in ids_arr), dtype=np.int64, count=n
        )
        keep = rows >= 0
        # -1 means "no landmark", which is not the same as a broken reference.
        n_dangling += int(((ids_arr >= 0) & ~keep).sum())
        if not keep.any():
            continue

        idx = np.flatnonzero(keep)
        obs_kf_l.append(np.full(len(idx), row, dtype=np.int64))
        obs_lm_l.append(rows[idx])
        obs_pt_l.append(np.asarray([kpts[i]["pt"] for i in idx], dtype=np.float64))
        obs_oct_l.append(np.asarray([kpts[i]["oct"] for i in idx], dtype=np.int8))
        obs_ang_l.append(np.asarray([kpts[i]["ang"] for i in idx], dtype=np.float32))
        obs_dep_l.append(depth[idx])
        obs_desc_l.append(_descs_to_bytes(v["descs"])[idx])

    def _cat(chunks, shape, dtype):
        return (
            np.concatenate(chunks)
            if chunks
            else np.zeros(shape, dtype=dtype)
        )

    obs_kf = _cat(obs_kf_l, (0,), np.int64)
    obs_lm = _cat(obs_lm_l, (0,), np.int64)
    obs_pt = _cat(obs_pt_l, (0, 2), np.float64)
    obs_octave = _cat(obs_oct_l, (0,), np.int8)
    obs_angle = _cat(obs_ang_l, (0,), np.float32)
    obs_depth = _cat(obs_dep_l, (0,), np.float32) * np.float32(scale)
    obs_desc = _cat(obs_desc_l, (0, 32), np.uint8)

    if n_dangling:
        warnings.warn(
            f"{path.name}: {n_dangling} keypoint(s) reference landmarks that are "
            f"not in the file; dropped",
            stacklevel=2,
        )

    return RawMap(
        path=path,
        cameras=cameras,
        orb_params=orb_params,
        kf_ids=kf_ids,
        kf_index=kf_index,
        kf_quat_cw=kf_quat_cw,
        kf_trans_cw=kf_trans_cw,
        kf_rot_cw=kf_rot_cw,
        kf_center=kf_center,
        kf_n_keypts=kf_n_keypts,
        kf_n_detected=kf_n_detected,
        kf_ts=kf_ts,
        kf_cam=kf_cam,
        kf_orb=kf_orb,
        kf_span_parent=kf_span_parent,
        kf_loop_edges=kf_loop_edges,
        lm_ids=lm_ids,
        lm_index=lm_index,
        lm_pos_w=lm_pos_w,
        lm_n_vis=lm_n_vis,
        lm_n_fnd=lm_n_fnd,
        lm_ref_kf=lm_ref_kf,
        lm_first_kf=lm_first_kf,
        obs_kf=obs_kf,
        obs_lm=obs_lm,
        obs_pt=obs_pt,
        obs_octave=obs_octave,
        obs_angle=obs_angle,
        obs_depth=obs_depth,
        obs_desc=obs_desc,
        n_dangling=n_dangling,
        scale=float(scale),
        scale_given=scale_given,
    )


def without_keyframes(raw: RawMap, drop_rows: np.ndarray) -> RawMap:
    """A copy of the map with some keyframes -- and what only they saw -- removed.

    Used to ask what the map would predict about a place it had never been:
    drop a scan station, rebuild, and see whether the prediction for that spot
    matches what the full map actually found there.

    Landmarks left with no observations go too, and every index is remapped so
    the result is indistinguishable from a map that was built that way.
    """
    drop = np.zeros(raw.n_keyframes, dtype=bool)
    drop[np.asarray(drop_rows, dtype=np.int64)] = True
    keep_kf = ~drop
    if not keep_kf.any():
        raise ValueError("cannot drop every keyframe")

    kf_new = np.full(raw.n_keyframes, -1, dtype=np.int64)
    kf_new[keep_kf] = np.arange(int(keep_kf.sum()))

    obs_keep = keep_kf[raw.obs_kf]
    obs_lm = raw.obs_lm[obs_keep]
    lm_alive = np.zeros(raw.n_landmarks, dtype=bool)
    lm_alive[obs_lm] = True
    lm_new = np.full(raw.n_landmarks, -1, dtype=np.int64)
    lm_new[lm_alive] = np.arange(int(lm_alive.sum()))

    kept_ids = set(raw.kf_ids[keep_kf].tolist())
    parent = raw.kf_span_parent[keep_kf].copy()
    parent[~np.isin(parent, list(kept_ids))] = -1

    return RawMap(
        path=raw.path,
        cameras=raw.cameras,
        orb_params=raw.orb_params,
        kf_ids=raw.kf_ids[keep_kf],
        kf_index={int(v): i for i, v in enumerate(raw.kf_ids[keep_kf])},
        kf_quat_cw=raw.kf_quat_cw[keep_kf],
        kf_trans_cw=raw.kf_trans_cw[keep_kf],
        kf_rot_cw=raw.kf_rot_cw[keep_kf],
        kf_center=raw.kf_center[keep_kf],
        kf_n_keypts=raw.kf_n_keypts[keep_kf],
        kf_n_detected=raw.kf_n_detected[keep_kf],
        kf_ts=raw.kf_ts[keep_kf],
        kf_cam=[c for c, k in zip(raw.kf_cam, keep_kf) if k],
        kf_orb=[o for o, k in zip(raw.kf_orb, keep_kf) if k],
        kf_span_parent=parent,
        kf_loop_edges=[[e for e in edges if e in kept_ids]
                       for edges, k in zip(raw.kf_loop_edges, keep_kf) if k],
        lm_ids=raw.lm_ids[lm_alive],
        lm_index={int(v): i for i, v in enumerate(raw.lm_ids[lm_alive])},
        lm_pos_w=raw.lm_pos_w[lm_alive],
        lm_n_vis=raw.lm_n_vis[lm_alive],
        lm_n_fnd=raw.lm_n_fnd[lm_alive],
        lm_ref_kf=raw.lm_ref_kf[lm_alive],
        lm_first_kf=raw.lm_first_kf[lm_alive],
        obs_kf=kf_new[raw.obs_kf[obs_keep]],
        obs_lm=lm_new[obs_lm],
        obs_pt=raw.obs_pt[obs_keep],
        obs_octave=raw.obs_octave[obs_keep],
        obs_angle=raw.obs_angle[obs_keep],
        obs_depth=raw.obs_depth[obs_keep],
        obs_desc=raw.obs_desc[obs_keep],
        n_dangling=raw.n_dangling,
        scale=raw.scale,
        scale_given=raw.scale_given,
    )
