# slammap2heatmap in Docker

Point it at a stella_vslam map (`.msg`) and it tells you, without anyone
walking around with a device, where in the building localization will work
and where it won't. You get a floor plan recovered from the map, heatmaps for
**tracking** (keeping hold of your position) and **relocalization** (finding
yourself from nothing), a combined "will it work here" layer, and suggestions
for where to scan next. Results open as a single HTML file in any browser.

Nothing to install except Docker.

## 1. Get the image

**If someone sent you a `.tar.gz` file:**

```bash
docker load -i slammap2heatmap-0.3.0.tar.gz
```

**If you have the source code:**

```bash
docker/build.sh            # builds slammap2heatmap:latest
docker/build.sh --save     # ...and writes a .tar.gz you can pass on
```

## 2. Run it

Use the `slammap` script in this folder. Copy it anywhere on your `PATH` if
you like; it only needs Docker.

```bash
./slammap inspect  map.msg                       # is this map readable, and what is in it?
./slammap heatmap  map.msg -o results --open     # build the heatmaps and open the viewer
./slammap heatmap  maps/ -o results --png all    # every map in a folder, plus report figures
./slammap validate map.msg                       # how far can these predictions be trusted?
./slammap suggest  map.msg -n 5                  # where to put the next scan position
./slammap validate-runs map.msg --frames frames.csv --fit --write-config cal.yaml
                                                 # calibrate on a real run (see the main README)
./slammap config > settings.yaml                 # every threshold, explained, to tune
./slammap heatmap  map.msg --config settings.yaml
```

Paths work exactly as you'd type them, relative or absolute. Your maps are
mounted read-only, and results are written as you, not as root.

Every command has `--help`.

### Without the script (Windows, or scripting)

Mount the folder holding your maps and run from inside it:

```bash
docker run --rm -v "$PWD:/data" -w /data --user "$(id -u):$(id -g)" \
    slammap2heatmap heatmap map.msg -o results
```

On Windows PowerShell, use `-v "${PWD}:/data"` and drop `--user`.

## 3. Look at the results

Open `results/<map>.html` in a browser. It is self-contained, so you can
e-mail it or put it on a share as a single file.

Start with **Keeps tracking**, the layer real runs agree with best. **Time
to find itself** shows relocalization. Layers marked *uncalibrated* haven't
been checked against a real run yet; pass a calibrated config with
`--config` to fix that. Click any
cell to see how much the answer depends on which way you're facing. Every
metric has a hover explanation; you don't need to know how the tool works to
read it.

With `--png`, you also get report-ready figures with title, scale bar,
colour scale and settings burned in.

## Which maps work

| Map | Works? |
|---|---|
| Built from a laser scan (CRISTAL) | Yes. This is what it was developed and validated on. |
| Ordinary stella_vslam run, RGBD or stereo | Yes. |
| Ordinary stella_vslam run, **monocular** | Yes, with `--scale` (see below). |
| Fisheye camera | Yes; the field of view is approximated. |
| 360° / equirectangular camera | `inspect` only. Scoring isn't supported yet. |
| Tiny object scans | They load, but every distance setting assumes room scale, so the results mean nothing. |

**Monocular maps have no real-world scale.** stella sets the scale from the
first few frames, so one "metre" in the map can be anything. Measure one
distance you know, for example a door width, in the map and in reality, then
pass `--scale <real metres / map units>`. If you already know the map is
metric, pass `--scale 1`. Without `--scale` the tool refuses rather than
producing confident nonsense.

**Hand-held runs** don't start level, so their world frame is tilted. The
tool measures gravity from the camera orientations and corrects for it;
`inspect` reports by how much.

## Things worth knowing before you trust a number

- The heatmap assumes the device doing the localizing has **the same camera
  the map was built with**. If your phone's field of view differs a lot from
  the mapping camera's, the numbers shift.
- It cannot see **look-alike places**. Two identical corridors will both look
  fine, even though a device would confuse them.
- Pass the stella config your device runs with `--stella-config`; the
  relocalizer thresholds come from it.
- Several thresholds were calibrated on six laser-scan maps. Run
  `./slammap validate` on yours. If it says **WEAK MAP**, the map itself is
  the problem, not the tool.

## Troubleshooting

**`permission denied ... docker.sock`**: your user can't talk to Docker.
Add yourself to the `docker` group (`sudo usermod -aG docker $USER`, then log
out and back in), or use Docker Desktop.

**`image 'slammap2heatmap:latest' not found`**: load or build the image
first (step 1).

**Apple Silicon Macs**: an image built on an Intel/AMD machine runs under
emulation. It works, just slower. Building it on the Mac itself
(`docker/build.sh`) gives a native one.

**Output files owned by root**: you ran `docker run` without `--user`. The
`slammap` script does this for you.
