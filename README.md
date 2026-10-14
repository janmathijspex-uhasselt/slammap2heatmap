# slammap_2_heatmap

Turn a [stella_vslam](https://github.com/stella-cv/stella_vslam) map database
(`.msg`) into a top-down picture of where a device will actually localize —
separately for **tracking** and **relocalization**, and combined.

The full design rationale, including what these particular maps turned out to
contain, is in [`docs/DESIGN_PLAN.html`](docs/DESIGN_PLAN.html) (open it in a
browser).

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"     # dev adds pytest and pyflakes
```

To share with someone who shouldn't have to install anything, see
[`docker/README.md`](docker/README.md): one `docker load`, then
`./slammap heatmap map.msg`.

## Usage

Everything goes through the `./run` wrapper:

```bash
./run inspect /path/to/map.msg          # one map
./run inspect /path/to/slam_maps/       # every .msg in a directory
./run inspect --help
```

It works from any directory and relative paths mean what you typed, so this
is fine:

```bash
cd ~/Werk/Datasets/DFL/slam_maps
~/Werk/slammap_2_heatmap/run inspect map_A105.msg
```

Use the wrapper rather than calling Python directly: it clears `PYTHONPATH`,
which a sourced ROS 2 workspace otherwise uses to push `/opt/ros/...` onto
`sys.path` even inside an activated virtualenv, where it can shadow packages
installed in `.venv`.

`inspect` decodes the map, verifies the geometry against the file's own
redundant data, and reports what the map contains. It exits non-zero if any
map fails the check.

```bash
./run heatmap /path/to/map.msg -o out --open
./run heatmap /path/to/slam_maps/ -o out      # every map
```

`heatmap` recovers the floor plan, scores every open cell from 16 headings,
and writes three files per map into `-o`:

| | |
|---|---|
| `<map>.html` | self-contained viewer — just open it, no server needed |
| `<map>.npz` | every raw array, for your own analysis |
| `<map>.json` | parameters and summary statistics |

Useful flags: `--res` (cell size, default 0.15 m), `--yaw-bins` (headings per
cell, default 16), `--height` (camera height to simulate, defaults to the
scanner's own), `--combine product|min|sum`, `--no-occlusion`, `--validate`,
`--open`.

```bash
./run validate /path/to/map.msg
```

`validate` checks the predictions against the only ground truth a single map
offers:

- **at its own keyframes** — standing where the scanner stood, facing where it
  faced, is the one pose where localization is known to succeed, so the model
  must say so. This is what calibrates `BOW_MATCH_YIELD`.
- **decay with distance** — quality must fall off away from the stations,
  monotonically, and where it falls off tells you how far apart stations can be.
- **leave one station out** — drop a station, rebuild the map without it, and
  ask the model to predict that spot blind.

Add `--no-holdout` to skip the slow third test.

```bash
./run suggest /path/to/map.msg -n 5
```

`suggest` ranks places to put the next scan station, greedily, so the second
suggestion does not restate the first from a metre away. It does not guess how
much a new station would help — it measures that from the map's own decay
curve, so a sparsely captured map predicts sparse help and says so.

Two limits, reported with the results: only cells known to be open floor can be
candidates, and the estimate ignores that a new station also creates map points
on surfaces no current one can see, so real improvement should be at least this
good.

`heatmap --suggest [N]` runs the same thing and draws the positions on the
viewer as orange diamonds.

## Calibrating on real runs

`validate` can only score the map against its own keyframe images, where
success is guaranteed and there is no gap between a rendered image and a real
camera frame. Real runs decide the numbers:

```bash
./run validate-runs map.msg --frames frames.csv \
    --stella-config stella.yaml --fit --write-config calibration/mine.yaml
./run heatmap map.msg --config calibration/mine.yaml
```

`frames.csv` is what the Qualisys alignment app writes: one row per video
frame, with ground-truth pose, stella's tracked/lost state, matched landmarks
and position error. `validate-runs` recomputes every layer at each frame's
true pose and asks each layer the question it claims to answer:

| Layer | Compared against |
|---|---|
| expected matched landmarks | landmarks stella matched |
| keeps tracking | frames already tracking: did the lock hold `horizon_s`? |
| relocalization (per attempt) | lost frames: did this one bring it back? |
| position uncertainty | 3D error against ground truth |

It reports AUC, reliability, ECE and Brier skill with 2 s block-bootstrap
intervals, then fits `matched_per_neff`, the tracking calibration,
`bow_match_yield` and the position floor, and writes them with provenance.
`--stella-config` takes the relocalizer thresholds from the config the device
actually runs and warns if its camera differs from the map's.

`calibration/dfl_neural_d455.yaml` is the first such fit: the DFL neural map
against two RealSense D455 runs. Treat it as a starting point for maps from
the same renderer and camera, not as universal. What it found is in
[`docs/FEEDBACK_response.md`](docs/FEEDBACK_response.md).

## Settings

Every threshold the scorers depend on lives in one file rather than in the
source. Print the annotated defaults and edit what you need:

```bash
./run config > my-scans.yaml
./run heatmap map.msg --config my-scans.yaml
```

`--config` works on `inspect`, `heatmap`, `validate` and `suggest`. Unknown
sections, misspelled settings, wrong types and mutually inconsistent values are
all rejected with a message naming the problem, rather than silently ignored.

Several defaults were measured rather than chosen — `bow_match_yield` in
particular is pinned by `validate`. Changing those means the numbers no longer
match what the validation was run against, so re-run `validate` after.

## Figures for reports

```bash
./run heatmap map.msg -o out --png                      # the combined layer
./run heatmap map.msg -o out --png reloc sigma_pos      # named layers
./run heatmap map.msg -o out --png all --suggest
```

PNGs carry their own title, scale bar, colour ramp with real values, scan
positions and a footer recording the settings, so they stand on their own in a
document. `--png-aggregate max|mean|min` picks how headings collapse and
`--png-scale` sets pixels per cell. The viewer's own **Save this view as PNG**
button produces the same thing for whatever is currently on screen.

## Which maps it accepts

Developed against maps the CRISTAL pipeline builds from a laser scan, but it
reads anything stock stella_vslam writes. Those differ in four ways, all
handled:

| | CRISTAL / laser scan | stock stella run |
|---|---|---|
| `lm_ids` | matched ones packed to the front | one per keypoint, `-1` interleaved |
| depths | from the scan, mm-accurate | sensor depth (RGBD/stereo), none at all (monocular) |
| world frame | gravity-aligned | the first camera's, tilted however it was held |
| scale | metric | metric, except monocular: arbitrary |
| `n_vis` / `n_fnd` | left at 1 | real, and used to down-weight unreliable landmarks |

**Monocular maps need `--scale`** (true metres / map units). Without it the
scoring commands refuse rather than produce confident nonsense; `inspect` still
works. **360° cameras** load and inspect, but aren't scored: the virtual
camera is the map's own, which would make every heading see everything.

`tests/test_generality.py` holds that claim to account. It rewrites a CRISTAL
map in stock form (interleaved `-1`, no depths, tilted 11° / -7° / 23°, scaled by
0.37) and checks the answer is unchanged. Untilted, it is bit-identical.
Tilted, the summary numbers move by a few percent. A pure rotation about
vertical moves them by the same amount, so that comes from rasterizing the
grid at a different angle, not from the gravity correction (recovered to
0.11°).

## Tests

```bash
./.venv/bin/python -m pytest
SLAMMAP_TEST_MAP=/path/to/other.msg ./.venv/bin/python -m pytest
```

They need one real map; they skip if the default isn't present.

## Status

| Milestone | |
|---|---|
| 1. Loader, map model, geometry self-check | done |
| 2. Occupancy grid and visibility profiles | done |
| 3. Tracking layer and viewer | done |
| 4. Relocalization and combined layers | done |
| 5a. Validation report | done |
| 5b. Next-scan-position suggester | done |
| 6. Config file and report-ready PNGs | done |
| 7. Any stock stella map; Docker image | done |
| 8. Calibration on real runs (`validate-runs`) | done |
| 9. 3D | not started |

## Layout

```
run              entry point, use this
docker/          image, build script, and the `slammap` wrapper for colleagues
calibration/     settings fitted on real runs, per map type and camera
tests/           generality tests against stock-format maps
pyproject.toml   installed into .venv in editable mode
slammap2heatmap/
  loader.py      decode .msg into typed arrays; absorbs every format quirk
  mapmodel.py    observation graph, normals, scale bands, stations, duplicates
  selfcheck.py   prove the decoding before anything is inferred from it
  grid.py        the ground-plane raster everything shares
  occupancy.py   floor plan, from synthetic laser scans
  visibility.py  how far you can see from each cell, per bearing
  config.py      every tunable threshold, and its rationale
  scorers.py     the localizability layers
  validate.py    check the predictions against the map's own ground truth
  runs.py        check and calibrate against real localization runs
  suggest.py     where to put the next scan station
  export.py      npz + json + the viewer
  render.py      report-ready PNGs
  viewer_template.html
  __main__.py    CLI
docs/            design plan
out/             generated heatmaps (gitignored)
```

## Verifying the decoding

The tool rests on one assumption: `rot_cw` is a quaternion in **(x, y, z, w)**
order and `trans_cw` its matching world→camera translation. That is checked
against redundant data in every file — reprojecting each landmark through a
keyframe that observed it must land on the stored keypoint, and the resulting
camera-frame z must equal the stored RGBD depth. On the six DFL maps this gives
a median error of 0.26–0.50 px and 1.2–5.1 mm. If a future map fails, the tool
says so instead of producing a confident, wrong picture.
