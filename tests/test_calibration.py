"""Calibration on real runs, and the settings that feed it.

The real-run tests use the first Qualisys check of the DFL neural map. They
are skipped where that data is absent, and pin two things: the metric code
reproduces an independent implementation's numbers exactly, and the fit lands
somewhere sane.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import poisson

from slammap2heatmap import runs
from slammap2heatmap.config import Config, ConfigError

ALIGN = Path.home() / "Werk/qualisys_laserscan_alignment"
FRAMES = ALIGN / ("projects/dfl_demo/heatmap_checks/"
                  "DFL_Qualisys_Laserscan_Alignment_dec9_neural/frames.csv")
STELLA = ALIGN / "projects/dfl_demo/slam_runs/rs_recording_1_localize/stella.yaml"
MAP = ALIGN / "example_data/.cache/DFL_Qualisys_Laserscan_Alignment_dec9_neural.msg"

real_runs = pytest.mark.skipif(
    not (FRAMES.exists() and MAP.exists()),
    reason="needs the DFL Qualisys check data")


# ---------------------------------------------------------------- settings

def test_old_config_files_still_load(tmp_path):
    f = tmp_path / "old.yaml"
    f.write_text("tracking:\n  rgbd_depth_coeff: 0.002\n"
                 "relocalization:\n  pnp_sharpness: 5\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Config.load(f)
    assert len(w) == 2 and "ignored" in str(w[0].message)


def test_stella_config_sets_relocalizer_gates(tmp_path):
    f = tmp_path / "stella.yaml"
    f.write_text("%YAML:1.0\nRelocalizer:\n  min_num_bow_matches: 15\n"
                 "  min_num_valid_obs: 20\n")
    c = Config()
    assert (c.relocalization.min_bow_matches, c.relocalization.min_valid_obs) == (20, 50)
    c.apply_stella(f)
    assert (c.relocalization.min_bow_matches, c.relocalization.min_valid_obs) == (15, 20)


def test_bad_pooling_is_rejected(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("relocalization:\n  pooling: average\n")
    with pytest.raises(ConfigError):
        Config.load(f)


def test_fitted_list_round_trips(tmp_path):
    c = Config()
    c.calibration.fitted = ["tracking", "relocalization"]
    c.calibration.source = "2026-10-07: test"
    f = tmp_path / "c.yaml"
    f.write_text(c.to_yaml())
    assert Config.load(f).to_dict() == c.to_dict()


def test_no_matches_means_no_relocalization():
    """The Poisson gate goes to zero; the logistic it replaced kept ~5 %."""
    assert poisson.sf(14, 0.0) == 0.0
    assert poisson.sf(14, 1.0) < 1e-9
    assert 0.4 < poisson.sf(14, 15.0) < 0.6


# ---------------------------------------------------------------- metrics

def test_auc_basics():
    y = np.array([0, 0, 1, 1])
    assert runs.auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert runs.auc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert runs.auc(np.array([0.5, 0.5, 0.5, 0.5]), y) == 0.5


@real_runs
def test_metrics_reproduce_the_independent_check():
    """Fed the predictions the first check used, the numbers must match it."""
    fr = runs.read_frames([FRAMES])
    rows = list(csv.DictReader(open(FRAMES)))

    def col(n):
        return np.array([float(r[f"pred_{n}"]) if r[f"pred_{n}"] else np.nan
                         for r in rows])

    L = {n: col(n) for n in ("track", "reloc", "combined", "n_eff",
                             "best_matches", "sigma_pos")}
    L.update(track_raw=L["track"], expected_matched=L["n_eff"],
             sigma_geom=L["sigma_pos"])
    at = runs.AtFrames(ok=np.isfinite(L["track"]), row=fr.their_cell,
                       hbin=fr.their_bin, flat=fr.their_cell, layers=L)
    c = runs.check(fr, at, horizon_s=1.0, n_boot=20).checks
    assert c["tracking"].extra["unconditional_auc"] == pytest.approx(0.8433, abs=1e-3)
    assert c["landmarks"].auc == pytest.approx(0.8526, abs=1e-3)
    assert c["landmarks"].rho == pytest.approx(0.6478, abs=1e-3)
    assert c["landmarks"].extra["ratio_median"] == pytest.approx(0.8917, abs=1e-3)
    assert c["position"].rho == pytest.approx(-0.0543, abs=1e-3)


@real_runs
def test_fit_on_real_runs(tmp_path):
    from slammap2heatmap.__main__ import main
    out = tmp_path / "fitted.yaml"
    code = main(["validate-runs", str(MAP), "--frames", str(FRAMES),
                 "--stella-config", str(STELLA), "--fit", "--bootstrap", "50",
                 "--write-config", str(out)])
    assert code == 0
    c = Config.load(out)
    assert set(c.calibration.fitted) == {"landmarks", "tracking",
                                         "relocalization", "position"}
    # real frames match the rendered map far worse than it matches itself
    assert 0.05 < c.relocalization.bow_match_yield < 0.5
    assert 0.8 < c.calibration.matched_per_neff < 1.0
    assert (c.relocalization.min_bow_matches, c.relocalization.min_valid_obs) == (15, 20)
