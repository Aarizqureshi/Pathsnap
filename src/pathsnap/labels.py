"""Turning the reference route into one label per GPS observation.

The ``.route`` file lists the arcs the vehicle traversed, in order, but it
does not say *which observation* was on *which arc*. We recover that link
with a monotone alignment (a dynamic-programming / DTW-style procedure):

    choose a non-decreasing route position j(i) for every observation i
    minimising  sum_i  dist(observation_i, route_arc[j(i)])

Because positions can only move forward along the route, an observation
can never be matched to a part of the route the vehicle has not reached
yet or has already left — which a naive "nearest route arc" rule gets
wrong at loops, U-turns and parallel two-way arcs.

The alignment cost is the only use of the route; the resulting labels are
reference labels, not verified ground truth. ``align_dist_m`` is kept per
observation so unreliable labels can be reported or filtered.
"""
from __future__ import annotations

import numpy as np

from .geo import project_to_segments
from .network import RoadNetwork


def align_route(x: np.ndarray, y: np.ndarray, route: np.ndarray, net: RoadNetwork,
                chunk: int = 2048):
    """Monotone alignment of observations (x, y) to the route arc sequence.

    Returns ``(route_pos, true_arc, align_dist_m)`` arrays of length len(x).
    """
    n, m = len(x), len(route)
    if n == 0 or m == 0:
        e = np.empty(0)
        return e.astype(np.int64), e.astype(np.int64), e
    ax, ay = net.ax[route], net.ay[route]
    bx, by = net.bx[route], net.by[route]

    back = np.empty((n, m), dtype=np.int32)
    cols = np.arange(m, dtype=np.int32)
    prev = None
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d, _ = project_to_segments(x[s:e, None], y[s:e, None], ax[None], ay[None], bx[None], by[None])
        for r in range(e - s):
            i = s + r
            if prev is None:
                cost = d[r].copy()
                back[i] = cols  # unused for the first row
            else:
                run_min = np.minimum.accumulate(prev)
                arg = np.maximum.accumulate(np.where(prev <= run_min, cols, 0))
                cost = d[r] + run_min
                back[i] = arg
            prev = cost

    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    true_arc = route[pos]
    dist, _ = project_to_segments(x, y, net.ax[true_arc], net.ay[true_arc],
                                  net.bx[true_arc], net.by[true_arc])
    return pos, true_arc, dist


def label_quality(align_dist: np.ndarray, route_pos: np.ndarray, route_len: int) -> dict:
    return {
        "align_dist_median_m": float(np.median(align_dist)),
        "align_dist_p95_m": float(np.percentile(align_dist, 95)),
        "align_dist_max_m": float(align_dist.max()),
        "frac_obs_over_20m": float(np.mean(align_dist > 20)),
        "frac_obs_over_50m": float(np.mean(align_dist > 50)),
        "route_arcs_used_frac": float(len(np.unique(route_pos)) / route_len),
    }
