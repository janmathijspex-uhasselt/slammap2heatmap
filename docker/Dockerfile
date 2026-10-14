# slammap2heatmap -- localizability heatmaps from stella_vslam maps.
#
# Build from the repository root (docker/build.sh does this):
#     docker build -f docker/Dockerfile -t slammap2heatmap .
FROM python:3.12-slim

LABEL org.opencontainers.image.title="slammap2heatmap" \
      org.opencontainers.image.description="Localizability heatmaps from stella_vslam map databases"

# Fonts for the report PNGs. Without them Pillow falls back to a tiny bitmap
# face and the figures are unreadable.
RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/slammap2heatmap
COPY pyproject.toml README.md ./
COPY slammap2heatmap ./slammap2heatmap
# A real (not editable) install, then prove the viewer template made it in:
# that file is package data, and forgetting to declare it is the one way this
# image can build cleanly and still be broken.
RUN pip install --no-cache-dir . \
 && python -c "import importlib.resources as r; \
assert r.files('slammap2heatmap').joinpath('viewer_template.html').is_file()" \
 && slammap2heatmap --help > /dev/null

# The container runs as the calling user (see docker/slammap), who cannot
# write to site-packages; stop Python trying.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /data
ENTRYPOINT ["slammap2heatmap"]
CMD ["--help"]
