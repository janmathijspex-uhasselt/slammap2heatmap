"""Derive the quantities the scorers need from a decoded map.

The map file stores raw geometry; localizability needs the things stella
computes at runtime and never writes down:

* the landmark -> observation graph, rebuilt from ``lm_ids`` because maps
  built from a laser scan leave the stored ``n_vis`` / ``n_fnd`` counters at 1
  (maps from an ordinary stella run do populate them, and they are used);
* a mean observation normal per landmark -- ORB cannot match a surface seen
  from the far side, so this bounds the viewing cone;
* the per-octave valid distance band, following stella's own
  ``landmark::compute_distance_invariance``;
* gravity and floor height, so a top-down grid can be laid out -- measured,
  because a hand-held run's world frame is tilted however its first camera
  was held;
* viewpoints: the scan stations of a panoramic scan, or short stretches of
  trajectory for a hand-held or robot run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .config import DEFAULT, Config
from .loader import Camera, OrbParams, RawMap

_AXES: dict[str, np.ndarray] = {
    "x": np.array([1.0, 0, 0]), "-x": np.array([-1.0, 0, 0]),
    "y": np.array([0, 1.0, 0]), "-y": np.array([0, -1.0, 0]),
    "z": np.array([0, 0, 1.0]), "-z": np.array([0, 0, -1.0]),
}


@dataclass
class Stations:
    """Viewpoints, and which keyframes were taken from each."""

    center: np.ndarray            # (S, 3) float64
    kf_rows: list[np.ndarray]     # per station, keyframe rows
    of_kf: np.ndarray             # (K,) int64, station index per keyframe row

    @property
    def n(self) -> int:
        return len(self.center)

    def spacing(self) -> np.ndarray:
        """Distance from each station to its nearest neighbour."""
        if self.n < 2:
            return np.zeros(self.n)
        tree = cKDTree(self.center)
        d, _ = tree.query(self.center, k=2)
        return d[:, 1]


@dataclass
class MapModel:
    """A map with everything the scorers need precomputed."""

    raw: RawMap
    camera: Camera
    orb: OrbParams

    # --- frame ------------------------------------------------------------
    up_axis: str                  # "z", "-y", or the measured vector if tilted
    up: np.ndarray                # (3,) unit vector, world "up"
    align: np.ndarray             # (3, 3) world -> (ground x, ground y, up)
    floor_h: float
    ceiling_h: float
    station_h: float

    # --- per landmark (row-aligned with raw.lm_pos_w) ----------------------
    n_obs: np.ndarray             # (L,) int32, rebuilt observation count
    n_stations: np.ndarray        # (L,) int32, distinct viewpoints observing it
    normal: np.ndarray            # (L, 3) float64, mean viewing direction
    d_min: np.ndarray             # (L,) float32, scale band lower bound
    d_max: np.ndarray             # (L,) float32, scale band upper bound
    d_ref: np.ndarray             # (L,) float32, reference viewing distance
    desc: np.ndarray              # (L, 32) uint8, representative descriptor
    ambiguity: np.ndarray         # (L,) float32, 0 = distinctive, 1 = ambiguous
    outlier: np.ndarray           # (L,) bool, inconsistent with its own observations
    dup_group: np.ndarray         # (L,) int64, shared by duplicate landmarks
    multiplicity: np.ndarray      # (L,) int32, size of the duplicate group
    weight: np.ndarray            # (L,) float32, probability of a good match
    counters_informative: bool    # whether n_vis / n_fnd carried a signal
    landmark_sigma: float         # per-axis map point error, metres
    landmark_sigma_measured: bool # True if it came from the map itself

    stations: Stations
    _obs_by_lm: tuple[np.ndarray, np.ndarray] = field(repr=False, default=None)

    # ------------------------------------------------------------------
    @property
    def n_landmarks(self) -> int:
        return len(self.n_obs)

    def observations_of(self, lm_row: int) -> np.ndarray:
        """Rows of the observation table belonging to one landmark."""
        starts, order = self._obs_by_lm
        return order[starts[lm_row]:starts[lm_row + 1]]

    def landmark_rows_of_kf(self, kf_row: int) -> np.ndarray:
        return self.raw.obs_lm[self.raw.obs_kf == kf_row]

    def ground(self, xyz: np.ndarray) -> np.ndarray:
        """World points or directions -> the two horizontal coordinates."""
        return np.asarray(xyz) @ self.align[:2].T

    def height(self, xyz: np.ndarray) -> np.ndarray:
        """World points or directions -> signed height along gravity."""
        return np.asarray(xyz) @ self.align[2]

    def heading(self, direction: np.ndarray) -> np.ndarray:
        """Compass heading, radians, of world directions in the ground plane."""
        g = self.ground(direction)
        return np.arctan2(g[..., 1], g[..., 0])


# ----------------------------------------------------------------------
def _detect_up(raw: RawMap) -> np.ndarray:
    """Infer gravity from where the cameras' own up vectors point.

    A camera's up in its own frame is -y (image y grows downward), so the
    world-frame up of keyframe k is ``R_cw.T @ (0, -1, 0)``.  Averaging over
    keyframes cancels the tilt of individual views and leaves gravity, as long
    as the camera was held roughly upright -- true of a scanner, a robot, and
    almost any hand-held capture.
    """
    up_w = -raw.kf_rot_cw[:, 1, :]          # row 1 of R_cw is the camera y axis
    mean = up_w.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 0.5:
        raise ValueError(
            "cannot infer the up axis: camera orientations are too varied "
            f"(mean up vector has length {norm:.2f}). Pass --up explicitly."
        )
    return mean / norm


def _alignment(up: np.ndarray) -> np.ndarray:
    """Right-handed rotation taking ``up`` to +z.

    The ground x axis is the world axis most nearly horizontal, so an
    axis-aligned map keeps its own coordinates: up=z gives the identity, and
    up=-y gives ground (x, z).
    """
    up = up / np.linalg.norm(up)
    e = np.eye(3)[int(np.argmin(np.abs(up)))]
    a = e - (e @ up) * up
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    return np.stack([a, b, up])


def _describe_up(up: np.ndarray) -> str:
    for name, axis in _AXES.items():
        if np.allclose(up, axis):
            return name
    return f"({up[0]:+.3f}, {up[1]:+.3f}, {up[2]:+.3f})"


def _cluster_stations(centers: np.ndarray, radius: float) -> Stations:
    """Group keyframes into viewpoints.

    For a panoramic scan this recovers the scan stations exactly: their
    keyframes share a camera centre.  For a hand-held or robot run it cuts the
    trajectory into stretches about ``radius`` long.  Leader clustering, in
    keyframe order, so a slow walk cannot chain the whole run into one group
    the way single-linkage would.
    """
    n = len(centers)
    of_kf = np.full(n, -1, dtype=np.int64)
    leaders: list[np.ndarray] = []
    for i in range(n):
        if leaders:
            d = np.linalg.norm(np.asarray(leaders) - centers[i], axis=1)
            j = int(np.argmin(d))
            if d[j] <= radius:
                of_kf[i] = j
                continue
        of_kf[i] = len(leaders)
        leaders.append(centers[i])
    S = len(leaders)
    station_center = np.stack([centers[of_kf == s].mean(axis=0) for s in range(S)])
    kf_rows = [np.flatnonzero(of_kf == s) for s in range(S)]
    return Stations(center=station_center, kf_rows=kf_rows, of_kf=of_kf)


_POPCOUNT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1
).sum(axis=1).astype(np.uint8)


def _hamming(a: np.ndarray, b: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Bitwise Hamming distance between broadcastable (..., 32) uint8 arrays.

    Chunked over the first axis so a full neighbour table does not have to be
    unpacked into memory at once.
    """
    a, b = np.broadcast_arrays(a, b)
    out = np.empty(a.shape[:-1], dtype=np.int16)
    for i in range(0, len(a), chunk):
        out[i:i + chunk] = _POPCOUNT[a[i:i + chunk] ^ b[i:i + chunk]].sum(
            axis=-1, dtype=np.int16
        )
    return out


def _duplicate_groups(pos: np.ndarray, desc: np.ndarray,
                      radius: float, max_hamming: int) -> np.ndarray:
    """Label landmarks that are the same physical feature seen twice.

    These maps have no loop closure, so stella's landmark fusion never ran:
    each station independently creates its own landmark for the same corner,
    and they survive side by side.  In the DFL scans 63% of landmarks have a
    neighbour within 5 cm, and 39% of those pairs are observed by completely
    disjoint sets of stations -- duplicates, not distinct structure.

    They matter because a single image keypoint can only ever match one of
    them.  Counting a cluster of three as three trackable points would
    overstate the map by up to 40% and wreck the calibration against stella's
    real match thresholds.

    Returns a group label per landmark; co-located look-alikes share one.
    """
    n = len(pos)
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = np.fromiter(
        (i for pair in cKDTree(pos).query_pairs(radius) for i in pair),
        dtype=np.int64,
    ).reshape(-1, 2)
    if len(pairs):
        close = _hamming(desc[pairs[:, 0]], desc[pairs[:, 1]]) <= max_hamming
        for a, b in pairs[close]:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    labels = np.array([find(i) for i in range(n)], dtype=np.int64)
    return np.unique(labels, return_inverse=True)[1]


def _ambiguity(pos: np.ndarray, desc: np.ndarray, r_inner: float,
               r_outer: float, k: int) -> np.ndarray:
    """How alike a landmark looks to the *other* points it could be mistaken for.

    Only neighbours in an annulus count.  Closer than ``r_inner`` a look-alike
    is either the same physical feature or so nearby that picking the wrong
    one costs centimetres -- harmless.  Beyond ``r_outer`` the two no longer
    fall in the same search window during tracking.  In between sits the real
    hazard: repetitive structure -- tiled floors, identical door frames,
    window mullions -- that yields confident wrong matches.

    Returns 0 for a distinctive point, 1 for one indistinguishable from a
    genuine alternative.
    """
    n = len(pos)
    if n < 3:
        return np.zeros(n, dtype=np.float32)
    k = min(k, n - 1)
    dist, idx = cKDTree(pos).query(pos, k=k + 1, distance_upper_bound=r_outer)
    dist, idx = dist[:, 1:], idx[:, 1:]                  # drop self
    valid = (dist >= r_inner) & (dist <= r_outer) & (idx < n)
    idx = np.where(valid, idx, 0)                        # clamp for gather

    h = _hamming(desc[:, None, :], desc[idx]).astype(np.float32)
    h[~valid] = np.inf
    closest = h.min(axis=1)
    # stella's ORB match thresholds are 50 bits (strict) and 100 (permissive)
    # out of 256.  Below 50 two points are effectively indistinguishable.
    amb = np.clip(1.0 - (closest - 50.0) / 50.0, 0.0, 1.0)
    return np.nan_to_num(amb, nan=0.0, posinf=0.0).astype(np.float32)


def _representative_descriptors(raw: RawMap, starts: np.ndarray,
                                order: np.ndarray) -> np.ndarray:
    """One descriptor per landmark: the observation closest to all the others.

    Mirrors ``landmark::compute_descriptor``.  With a median of two
    observations per landmark the choice rarely matters, but it does for the
    well-observed points that carry the most weight.
    """
    L = len(starts) - 1
    out = np.zeros((L, 32), dtype=np.uint8)
    desc = raw.obs_desc
    for row in range(L):
        sel = order[starts[row]:starts[row + 1]]
        if len(sel) == 0:
            continue
        if len(sel) <= 2:
            out[row] = desc[sel[0]]
            continue
        block = desc[sel]
        d = _hamming(block[:, None, :], block[None, :, :])
        out[row] = block[int(np.argmin(np.median(d, axis=1)))]
    return out


LANDMARK_SIGMA_FALLBACK_M = 0.03
"""Used when a map has too few duplicate landmarks to measure from."""


def _landmark_sigma(pos: np.ndarray, desc: np.ndarray, viewpoint: np.ndarray,
                    radius: float = 0.40, k: int = 32,
                    min_pairs: int = 50) -> tuple[float, int]:
    """Per-axis map point error, measured from the map's own duplicates.

    Where two viewpoints each placed a landmark for the same physical corner,
    the gap between them is two independent placement errors.  For an
    isotropic 3D Gaussian with per-axis sigma, the median distance between
    two such points is ``1.538 * sqrt(2) * sigma``, about ``2.175 * sigma``.

    Pairs are *mutual best descriptor matches* among spatial neighbours seen
    from different viewpoints.  Both obvious shortcuts are biased: taking only
    the nearest few neighbours drops duplicates that landed further apart
    (too low), and taking every look-alike within the radius admits chance
    matches (too high).  With mutual matching the estimate stops moving as
    the search widens -- k=32 and k=64 agree within 1 mm on the maps tested.

    This captures whatever made the map: laser scan registration, rendered
    depth, bundle adjustment.  It does *not* capture error shared by both
    copies, such as the whole map being offset from the world; that is what
    the systematic floor is for.

    Returns (sigma, number of pairs used).
    """
    n = len(pos)
    if n < 3:
        return LANDMARK_SIGMA_FALLBACK_M, 0
    kk = min(k, n - 1)
    dist, idx = cKDTree(pos).query(pos, k=kk + 1, distance_upper_bound=radius)
    dist, idx = dist[:, 1:], idx[:, 1:]
    valid = idx < n
    nb = np.where(valid, idx, 0)

    h = _hamming(desc[:, None, :], desc[nb]).astype(np.float32)
    h[~valid] = np.inf
    h[viewpoint[nb] == viewpoint[:, None]] = np.inf          # same viewpoint
    h[(viewpoint[:, None] < 0) | (viewpoint[nb] < 0)] = np.inf

    rows = np.arange(n)
    best = np.argmin(h, axis=1)
    best_h = h[rows, best]
    partner = nb[rows, best]
    good = best_h < 50                                       # stella's strict match
    mutual = good & good[partner] & (nb[partner, best[partner]] == rows)
    pair = mutual & (rows < partner)                         # each pair once
    if pair.sum() < min_pairs:
        return LANDMARK_SIGMA_FALLBACK_M, int(pair.sum())
    return float(np.median(dist[rows, best][pair]) / 2.175), int(pair.sum())


def _outliers(raw: RawMap, camera: Camera, max_reproj: float) -> np.ndarray:
    """Landmarks that contradict the keyframes that observed them.

    Every map in the DFL set carries a few: points behind a camera that
    supposedly saw them, or reprojecting thousands of pixels away.  They are
    map defects rather than decoding errors, and letting them into the grid
    would put phantom structure in places nothing was ever scanned.
    """
    bad = np.zeros(raw.n_landmarks, dtype=bool)
    if raw.n_observations == 0:
        return bad
    pc = (
        np.einsum("nij,nj->ni", raw.kf_rot_cw[raw.obs_kf], raw.lm_pos_w[raw.obs_lm])
        + raw.kf_trans_cw[raw.obs_kf]
    )
    uv, front = camera.project(pc)
    behind = ~front
    err = np.full(len(pc), np.inf)
    err[front] = np.linalg.norm(uv[front] - raw.obs_pt[front], axis=1)
    if camera.is_spherical:
        du = np.abs(uv[front, 0] - raw.obs_pt[front, 0])
        du = np.minimum(du, camera.cols - du)
        err[front] = np.hypot(du, uv[front, 1] - raw.obs_pt[front, 1])
    np.logical_or.at(bad, raw.obs_lm, behind | (err > max_reproj))
    return bad


def build(raw: RawMap, up_axis: str | None = None,
          cfg: Config | None = None, up: np.ndarray | None = None) -> MapModel:
    """Precompute everything the scorers need from a decoded map.

    Gravity comes from, in order: ``up`` (a vector inherited from a parent
    map, so a reduced copy lands on exactly the same grid), ``up_axis`` (a
    named axis the user vouched for), or measurement.
    """
    conf = cfg or DEFAULT
    c = conf.map
    camera = raw.camera
    orb = raw.orb
    L = raw.n_landmarks

    # --- observation index: landmark row -> its observation rows -----------
    order = np.argsort(raw.obs_lm, kind="stable")
    counts = np.bincount(raw.obs_lm, minlength=L)
    starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    n_obs = counts.astype(np.int32)

    # --- mean observation normal ------------------------------------------
    # Direction from landmark towards each observing camera, averaged.
    rays = raw.kf_center[raw.obs_kf] - raw.lm_pos_w[raw.obs_lm]
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-9)
    normal = np.zeros((L, 3), dtype=np.float64)
    np.add.at(normal, raw.obs_lm, rays)
    nrm = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.divide(normal, nrm, out=np.zeros_like(normal), where=nrm > 1e-9)

    # --- scale band, following landmark::compute_distance_invariance -------
    # Use the reference keyframe's observation where present, else the first.
    ref_obs = np.full(L, -1, dtype=np.int64)
    ref_obs[raw.obs_lm[order[starts[:-1][counts > 0]]]] = order[starts[:-1][counts > 0]]
    ref_kf_row = np.array(
        [raw.kf_index.get(int(i), -1) for i in raw.lm_ref_kf], dtype=np.int64
    )
    for row in np.flatnonzero((ref_kf_row >= 0) & (counts > 0)):
        sel = order[starts[row]:starts[row + 1]]
        hit = sel[raw.obs_kf[sel] == ref_kf_row[row]]
        if len(hit):
            ref_obs[row] = hit[0]

    has = ref_obs >= 0
    scale_factors = orb.scale_factors
    d_ref = np.zeros(L, dtype=np.float32)
    d_ref[has] = np.linalg.norm(
        raw.lm_pos_w[has] - raw.kf_center[raw.obs_kf[ref_obs[has]]], axis=1
    )
    level = np.zeros(L, dtype=np.int64)
    level[has] = raw.obs_octave[ref_obs[has]].astype(np.int64)
    d_max = (d_ref * scale_factors[level]).astype(np.float32)
    d_min = (d_max / scale_factors[orb.num_levels - 1]).astype(np.float32)

    stations = _cluster_stations(raw.kf_center, c.viewpoint_radius_m)

    # --- descriptors and weights ------------------------------------------
    desc = _representative_descriptors(raw, starts, order)
    ambiguity = _ambiguity(raw.lm_pos_w, desc, c.confusion_min_m,
                           c.confusion_max_m, c.desc_neighbours)
    dup_group = _duplicate_groups(raw.lm_pos_w, desc,
                                  c.duplicate_radius_m, c.duplicate_hamming)
    outlier = _outliers(raw, camera, c.outlier_reproj_px)
    multiplicity = np.bincount(dup_group)[dup_group].astype(np.int32)

    # NOTE on what is deliberately absent: the landmark's own depth
    # uncertainty. It does not belong here. Range error makes a point's
    # *position* vague, not its descriptor unmatchable, and it already enters
    # the scorers through the Fisher information term. Applying it in both
    # places charged far landmarks twice for one defect.
    #
    # Repeatability is about distinct *viewpoints*, not keyframe count: the
    # two pitch views of one station share an optical centre, and consecutive
    # frames of a slow walk are centimetres apart. Neither proves anything
    # about how the feature holds up when you move.
    st_of_obs = stations.of_kf[raw.obs_kf]
    n_stations = np.zeros(L, dtype=np.int32)
    seen: set[tuple[int, int]] = set()
    for lm, st in zip(raw.obs_lm.tolist(), st_of_obs.tolist()):
        if (lm, st) not in seen:
            seen.add((lm, st))
            n_stations[lm] += 1
    q_obs = 1.0 - 0.5 ** np.maximum(n_stations.astype(np.float32), 0.0)
    # Repetitive structure yields confident wrong matches.
    q_desc = 1.0 - ambiguity
    # A duplicate cluster can only ever satisfy one image keypoint.
    q_unique = 1.0 / multiplicity.astype(np.float32)
    # Maps from an ordinary stella run carry real visibility counters; scan-
    # built maps leave them at 1. Where they mean something, use them: a
    # landmark the tracker kept failing to find is not worth counting on.
    counters_informative = bool((raw.lm_n_vis > 1).any())
    if counters_informative:
        ratio = raw.lm_n_fnd / np.maximum(raw.lm_n_vis, 1)
        q_found = np.where(raw.lm_n_vis >= 3,
                           np.clip(ratio / c.found_ratio_full, 0.0, 1.0), 1.0)
    else:
        q_found = np.ones(L)
    weight = (q_obs * q_desc * q_unique * q_found).astype(np.float32)
    weight[~has] = 0.0
    weight[outlier] = 0.0

    # --- how accurately the map placed its points --------------------------
    if conf.tracking.landmark_sigma_m > 0:
        landmark_sigma, measured = float(conf.tracking.landmark_sigma_m), False
    else:
        viewpoint = np.full(L, -1, dtype=np.int64)
        viewpoint[has] = stations.of_kf[raw.obs_kf[ref_obs[has]]]
        landmark_sigma, n_pairs = _landmark_sigma(raw.lm_pos_w, desc, viewpoint)
        measured = n_pairs >= 50

    # --- frame ------------------------------------------------------------
    if up is not None:
        up = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    elif up_axis is not None:
        if up_axis not in _AXES:
            raise ValueError(
                f"--up must be one of {', '.join(_AXES)}, got {up_axis!r}"
            )
        up = _AXES[up_axis]
    else:
        up = _detect_up(raw)
        nearest = max(_AXES.values(), key=lambda a: float(a @ up))
        if np.degrees(np.arccos(np.clip(nearest @ up, -1, 1))) <= c.up_snap_deg:
            up = nearest
    align = _alignment(up)

    lm_h = raw.lm_pos_w @ align[2]
    floor_h, ceiling_h = np.percentile(lm_h, [2.0, 98.0])
    station_h = float(np.median(raw.kf_center @ align[2]))

    return MapModel(
        raw=raw, camera=camera, orb=orb,
        up_axis=_describe_up(up), up=up, align=align,
        floor_h=float(floor_h), ceiling_h=float(ceiling_h), station_h=station_h,
        n_obs=n_obs, n_stations=n_stations, normal=normal,
        d_min=d_min, d_max=d_max, d_ref=d_ref,
        desc=desc, ambiguity=ambiguity, outlier=outlier, dup_group=dup_group,
        multiplicity=multiplicity, weight=weight,
        counters_informative=counters_informative,
        landmark_sigma=landmark_sigma, landmark_sigma_measured=measured,
        stations=stations, _obs_by_lm=(starts, order),
    )
