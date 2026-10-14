"""The 2D ground-plane grid every raster in this tool shares."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mapmodel import MapModel


@dataclass(frozen=True)
class Grid:
    """A regular raster over the two horizontal world axes.

    Cell (0, 0) is at ``origin``; ``(ix, iy)`` covers world
    ``origin + (ix, iy) * res`` to ``origin + (ix + 1, iy + 1) * res``.
    Values are sampled at cell centres.
    """

    origin: np.ndarray        # (2,) world coordinates of the grid corner
    res: float                # metres per cell
    nx: int
    ny: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.nx, self.ny

    @property
    def size(self) -> int:
        return self.nx * self.ny

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)`` in world units, for plotting."""
        return (
            float(self.origin[0]),
            float(self.origin[1]),
            float(self.origin[0] + self.nx * self.res),
            float(self.origin[1] + self.ny * self.res),
        )

    def centers(self) -> np.ndarray:
        """(nx * ny, 2) world coordinates of every cell centre, row-major in x."""
        xs = self.origin[0] + (np.arange(self.nx) + 0.5) * self.res
        ys = self.origin[1] + (np.arange(self.ny) + 0.5) * self.res
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        return np.column_stack([gx.ravel(), gy.ravel()])

    def to_cell(self, xy: np.ndarray) -> np.ndarray:
        """World coordinates -> integer cell indices (may fall outside)."""
        return np.floor((np.asarray(xy) - self.origin) / self.res).astype(np.int64)

    def flat_index(self, ij: np.ndarray) -> np.ndarray:
        """Cell indices -> flat index, or -1 where outside the grid."""
        ij = np.asarray(ij)
        inside = (
            (ij[..., 0] >= 0) & (ij[..., 0] < self.nx)
            & (ij[..., 1] >= 0) & (ij[..., 1] < self.ny)
        )
        return np.where(inside, ij[..., 0] * self.ny + ij[..., 1], -1)

    def sample_flat(self, xy: np.ndarray) -> np.ndarray:
        """World coordinates -> flat cell index, -1 outside."""
        return self.flat_index(self.to_cell(xy))


def fit(model: MapModel, res: float, margin: float = 1.5,
        percentile: float = 1.0) -> Grid:
    """Lay out a grid covering the map, robust to stray landmarks.

    Landmark bounding boxes in these maps are wrecked by outliers -- one scan
    spans 97 m at min/max but 37 m between the 1st and 99th percentile -- so
    the extent comes from percentiles, then is widened to include every
    station and a working margin.
    """
    g = model.ground(model.raw.lm_pos_w[~model.outlier])
    lo = np.percentile(g, percentile, axis=0)
    hi = np.percentile(g, 100.0 - percentile, axis=0)

    st = model.ground(model.stations.center)
    lo = np.minimum(lo, st.min(axis=0)) - margin
    hi = np.maximum(hi, st.max(axis=0)) + margin

    origin = np.floor(lo / res) * res
    nx, ny = np.ceil((hi - origin) / res).astype(int) + 1
    return Grid(origin=origin, res=float(res), nx=int(nx), ny=int(ny))
