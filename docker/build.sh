#!/usr/bin/env bash
# Build the image. With --save, also package it as a single file that can be
# handed to someone without a registry:
#
#     docker/build.sh --save        ->  docker/slammap2heatmap-<version>.tar.gz
#
# They load it with:  docker load -i slammap2heatmap-<version>.tar.gz
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
version="$(sed -n 's/^version = "\(.*\)"/\1/p' "$root/pyproject.toml" | head -1)"

docker build -f "$here/Dockerfile" \
    -t "slammap2heatmap:$version" -t "slammap2heatmap:latest" "$root"
echo "built slammap2heatmap:$version (also tagged latest)"

if [[ "${1:-}" == "--save" ]]; then
    out="$here/slammap2heatmap-$version.tar.gz"
    docker save "slammap2heatmap:$version" "slammap2heatmap:latest" | gzip > "$out"
    echo "wrote $out ($(du -h "$out" | cut -f1)) -- this is the file to share"
fi
