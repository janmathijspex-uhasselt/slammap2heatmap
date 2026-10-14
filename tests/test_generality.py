"""Does the tool give the same answer for a map written the way stock stella writes it?

Every real map tested so far was built by the CRISTAL pipeline from a laser
scan: gravity-aligned, metric, with depths, and with ``lm_ids`` packed.  A map
from an ordinary hand-held stella run differs in four ways at once:

* ``lm_ids`` holds one entry per keypoint, with -1 interleaved;
* a monocular run stores no depths at all;
* the world frame is the first camera's, tilted however it was held;
* a monocular map has arbitrary scale.

There is no such map on hand with ground truth, so these tests make one: take
a scan-built map and rewrite it in exactly that form.  The geometry is
unchanged -- only how it is expressed -- so a correct tool, told the scale,
must give the same answer.  Anything that leaks the original conventions into
the result shows up as a difference.

    pytest tests/                       # uses the default map below
    SLAMMAP_TEST_MAP=other.msg pytest tests/
"""

from __future__ import annotations

import os
from pathlib import Path

import msgpack
import numpy as np
import pytest

from slammap2heatmap import grid as gridmod
from slammap2heatmap import loader, mapmodel, occupancy, scorers, selfcheck, visibility

DEFAULT_MAP = "/home/pex/Werk/Datasets/DFL/map_xr_exp_room.msg"
MAP = Path(os.environ.get("SLAMMAP_TEST_MAP", DEFAULT_MAP))

pytestmark = pytest.mark.skipif(
    not MAP.exists(),
    reason=f"test map {MAP} not found; set SLAMMAP_TEST_MAP to any .msg file",
)

SCALE = 0.37                         # map units per metre in the rewritten map
TILT = np.radians([11.0, -7.0, 23.0])  # roll, pitch, yaw applied to the world


def _rot(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = (np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry),
                              np.cos(rz), np.sin(rz))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _quat_xyzw(R: np.ndarray) -> list[float]:
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    return [float(x), float(y), float(z), float(w)]


def _as_stock_monocular(src: Path, dst: Path, rng: np.random.Generator,
                        tilt=TILT) -> None:
    """Rewrite a scan-built map as a hand-held monocular stella run would save it."""
    blob = msgpack.unpackb(src.read_bytes(), raw=False, strict_map_key=False)
    Rt = _rot(*tilt)

    for cam in blob["cameras"].values():
        cam["setup_type"] = "Monocular"
        cam["focal_x_baseline"] = 0.0

    for lm in blob["landmarks"].values():
        lm["pos_w"] = (SCALE * (Rt @ np.asarray(lm["pos_w"]))).tolist()

    for kf in blob["keyframes"].values():
        # x_c = R x_w + t  and  x_w = Rt^T x_w'  ->  R' = R Rt^T,  t' = s t
        q = np.asarray(kf["rot_cw"], dtype=np.float64)
        R = loader.quats_to_rotations(q[None])[0]
        kf["rot_cw"] = _quat_xyzw(R @ Rt.T)
        kf["trans_cw"] = (SCALE * np.asarray(kf["trans_cw"])).tolist()

        # Stock layout: one lm_id per keypoint, with unmatched keypoints
        # interleaved as -1 rather than packed to the end.
        n = kf["n_keypts"]
        ids, pts, descs = kf["lm_ids"][:n], kf["undist_keypts"], kf["descs"]
        extra = int(rng.integers(n // 2, n + 1)) if n else 0
        order = rng.permutation(n + extra)
        new_ids, new_pts, new_descs = [None] * (n + extra), [None] * (n + extra), [None] * (n + extra)
        for slot, k in zip(order, range(n + extra)):
            if k < n:
                new_ids[slot], new_pts[slot], new_descs[slot] = ids[k], pts[k], descs[k]
            else:
                new_ids[slot] = -1
                new_pts[slot] = {"ang": 0.0, "oct": 0,
                                 "pt": [float(rng.uniform(0, 1280)), float(rng.uniform(0, 720))]}
                new_descs[slot] = [int(v) for v in rng.integers(0, 2**32, 8, dtype=np.uint64)]
        kf["lm_ids"], kf["undist_keypts"], kf["descs"] = new_ids, new_pts, new_descs
        kf["n_keypts"] = n + extra
        kf["depths"] = []                    # monocular: nothing stored
        kf["x_rights"] = []

    dst.write_bytes(msgpack.packb(blob, use_bin_type=True))


def _score(path: Path, scale: float | None):
    raw = loader.load(path, scale=scale)
    model = mapmodel.build(raw)
    grid = gridmod.fit(model, res=0.15)
    occ = occupancy.build(model, grid)
    cells = np.flatnonzero(occ.free.ravel())
    vis = visibility.sweep(occ, cells)
    res = scorers.evaluate(model, grid, occ, vis, height=model.station_h)
    return raw, model, occ, res


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The original, plus three rewrites of it.

    * ``flat``    -- stock layout, monocular, rescaled, but not tilted.
                     Nothing about the geometry changes, so the answer must
                     be bit-identical.
    * ``tilted``  -- as above, plus the world frame rolled, pitched and
                     yawed, as a hand-held run's would be.
    * ``spun``    -- the control: rotated about vertical only, by exactly the
                     angle the tilted map's grid ends up at.  No gravity is
                     estimated, so whatever this changes is purely the cost of
                     laying the grid at that angle.  Matching the angle
                     matters: rasterization error is not monotonic in it, and
                     on the default map ranges from 1% to 5% over a few
                     degrees.
    """
    d = tmp_path_factory.mktemp("stock")
    rng = np.random.default_rng
    out = {"original": _score(MAP, None)}
    for name, tilt in (("flat", np.zeros(3)), ("tilted", TILT)):
        path = d / f"{name}.msg"
        _as_stock_monocular(MAP, path, rng(0), tilt=tilt)
        out[name] = _score(path, 1.0 / SCALE)
        out[name + "_path"] = path

    # Where did the tilted map's ground x axis land, relative to the original's?
    a0 = _rot(*TILT) @ out["original"][1].align[0]
    m1 = out["tilted"][1]
    spin = np.arctan2(m1.align[1] @ a0, m1.align[0] @ a0)
    path = d / "spun.msg"
    _as_stock_monocular(MAP, path, rng(0), tilt=np.array([0.0, 0.0, spin]))
    out["spun"] = _score(path, 1.0 / SCALE)
    return out


def test_stock_layout_is_not_reported_as_dangling(runs):
    raw = runs["tilted"][0]
    assert raw.n_dangling == 0, (
        "-1 entries in lm_ids mean 'no landmark' and must not be counted as "
        "broken references"
    )


def test_monocular_map_without_depth_passes_geometry_check(runs):
    raw = runs["tilted"][0]
    check = selfcheck.check_geometry(raw)
    assert not raw.has_depth
    assert check.ok, check.describe()


def test_monocular_without_scale_is_flagged_not_metric(runs):
    path = runs["tilted_path"]
    assert not loader.load(path).is_metric
    assert loader.load(path, scale=1.0).is_metric


def test_untilted_rewrite_is_bit_identical(runs):
    """Layout, missing depths and rescaling must change nothing at all."""
    (_, _, o0, r0), (_, _, o1, r1) = runs["original"], runs["flat"]
    assert np.array_equal(o0.state, o1.state)
    for name in ("n_eff", "track", "reloc", "combined"):
        np.testing.assert_allclose(getattr(r1, name), getattr(r0, name),
                                   rtol=1e-4, atol=1e-5, err_msg=name)


def test_tilted_frame_recovers_gravity(runs):
    m0, m1 = runs["original"][1], runs["tilted"][1]
    expected = _rot(*TILT) @ m0.up
    angle = np.degrees(np.arccos(np.clip(abs(expected @ m1.up), -1, 1)))
    assert angle < 0.5, f"gravity off by {angle:.2f} deg"
    assert m1.up_axis not in ("x", "y", "z", "-x", "-y", "-z")


def test_tilted_geometry_survives(runs):
    m0, m1 = runs["original"][1], runs["tilted"][1]
    assert m1.stations.n == m0.stations.n
    assert np.isclose(np.median(m1.d_ref), np.median(m0.d_ref), rtol=1e-6)
    assert abs((m1.ceiling_h - m1.floor_h) - (m0.ceiling_h - m0.floor_h)) < 0.05


def test_tilt_costs_no_more_than_rotating_the_grid(runs):
    """Gravity handling must add nothing beyond rasterization.

    Laying the grid at a different angle changes which cells exist near the
    walls, and that alone moves the summary numbers by a few percent.  The
    tilted map must not move them meaningfully further than the control,
    which only rotates and never estimates gravity.
    """
    r0 = runs["original"][3]
    # How far rasterization alone may move a layer. Counts move least.
    # Relocalization goes through steep Poisson thresholds, which amplify a
    # few percent fewer landmarks near a wall into roughly double that in
    # probability -- a property of thresholds, not of the tilt handling.
    raster_cap = {"n_eff": 0.08, "combined": 0.15}
    for name in ("n_eff", "combined"):
        base = np.median(getattr(r0, name).mean(axis=1))
        spun = np.median(getattr(runs["spun"][3], name).mean(axis=1))
        tilt = np.median(getattr(runs["tilted"][3], name).mean(axis=1))
        err_spun, err_tilt = abs(spun - base) / base, abs(tilt - base) / base
        assert err_spun < raster_cap[name], (
            f"{name}: rasterization alone moved it {err_spun:.1%}")
        assert err_tilt <= err_spun + 0.02, (
            f"{name}: tilt moved it {err_tilt:.1%}, a pure rotation only "
            f"{err_spun:.1%} -- gravity handling is adding error"
        )


def test_cli_refuses_unscaled_monocular(runs, tmp_path, capsys):
    from slammap2heatmap.__main__ import main
    code = main(["heatmap", str(runs["tilted_path"]), "-o", str(tmp_path)])
    assert code != 0
    assert "--scale" in capsys.readouterr().err
    assert not list(tmp_path.glob("*.html")), "must not write a heatmap it cannot stand behind"
    # ...and the same map, told its scale, goes through.
    assert main(["heatmap", str(runs["tilted_path"]), "-o", str(tmp_path),
                 "--scale", str(1.0 / SCALE)]) == 0


def test_cli_refuses_to_score_360_maps(tmp_path, capsys):
    """Loads and inspects, but will not pretend a phone sees in every direction."""
    from slammap2heatmap.__main__ import main
    blob = msgpack.unpackb(MAP.read_bytes(), raw=False, strict_map_key=False)
    for cam in blob["cameras"].values():
        cam.update(model_type="Equirectangular", setup_type="Monocular",
                   cols=2048, rows=1024)
        for k in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2"):
            cam.pop(k, None)
    path = tmp_path / "sphere.msg"
    path.write_bytes(msgpack.packb(blob, use_bin_type=True))

    raw = loader.load(path, scale=1.0)
    assert raw.camera.hfov_deg == 360.0
    assert main(["heatmap", str(path), "-o", str(tmp_path), "--scale", "1"]) != 0
    assert "360-degree" in capsys.readouterr().err
