"""Geodesy helpers.

The 100 trips are spread across the globe (Europe, Asia, the Americas,
Africa), so a single projected CRS cannot be used. Each trip gets its own
local East-North (tangent-plane) projection centred on the trip. For the
extents involved (tens of km) the orthographic tangent-plane distortion is
~(d/R)^2/6, i.e. < 0.01 % at 50 km, so planar point-to-segment geometry in
metres is accurate. ``scripts/01_dataset_inventory.py`` verifies this
against haversine distances.
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (vectorised)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class LocalProjection:
    """Orthographic projection onto the plane tangent to the sphere at (lon0, lat0).

    x points east, y points north, both in metres.
    """

    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self._l0 = np.radians(self.lon0)
        self._p0 = np.radians(self.lat0)

    @classmethod
    def from_points(cls, lon, lat) -> "LocalProjection":
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        # median is robust to a few wild GPS fixes
        return cls(np.median(lon), np.median(lat))

    def to_xy(self, lon, lat):
        lam = np.radians(np.asarray(lon, dtype=float))
        phi = np.radians(np.asarray(lat, dtype=float))
        dlam = lam - self._l0
        x = EARTH_RADIUS_M * np.cos(phi) * np.sin(dlam)
        y = EARTH_RADIUS_M * (
            np.cos(self._p0) * np.sin(phi) - np.sin(self._p0) * np.cos(phi) * np.cos(dlam)
        )
        return x, y


def project_to_segments(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB (all arrays broadcast together).

    Returns ``(dist, t)`` where ``t`` in [0, 1] is the position of the closest
    point along the segment (0 = A, 1 = B).
    """
    dx = bx - ax
    dy = by - ay
    seg_len2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
    t = np.where(seg_len2 > 0, t, 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx = ax + t * dx
    cy = ay + t * dy
    return np.hypot(px - cx, py - cy), t


def bearing_deg(dx, dy):
    """Compass bearing of a planar vector, degrees clockwise from north in [0, 360)."""
    return np.mod(np.degrees(np.arctan2(dx, dy)), 360.0)


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs(np.mod(np.asarray(a) - np.asarray(b), 360.0))
    return np.minimum(d, 360.0 - d)
