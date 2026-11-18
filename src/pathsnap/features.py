"""Per-candidate feature engineering.

One row per (observation, candidate arc). Features fall into four groups:

Geometric (how close is the observation to the arc?)
    dist_m, dist_rank, dist_minus_best_m, n_candidates, off_segment, arc_length_m
Directional (does the vehicle move the way the arc points?)
    heading_diff_deg, heading_valid
    -> the only feature that separates the two directed arcs of a two-way street
Temporal support (do the neighbouring fixes agree?)
    prev_dist_m, next_dist_m, context_mean_dist_m
Kinematic / topological context
    speed_ms, turn_rate_dps, dt_prev_s, end_out_degree

All features are computed from the observation sequence and the road
network only; the reference route is used solely to create ``label``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .geo import angle_diff_deg, project_to_segments
from .network import RoadNetwork

NEUTRAL_HEADING_DIFF = 90.0  # used when heading is unreliable (vehicle ~stationary)


def _neighbour_dist(points: pd.DataFrame, p: np.ndarray, a: np.ndarray, offset: int,
                    net: RoadNetwork, own_dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from observation p+offset to arc a (same segment only)."""
    n = len(points)
    q = p + offset
    seg = points["segment_id"].to_numpy()
    ok = (q >= 0) & (q < n)
    ok[ok] &= seg[q[ok]] == seg[p[ok]]
    d = own_dist.copy()
    if ok.any():
        qx = points["x"].to_numpy()[q[ok]]
        qy = points["y"].to_numpy()[q[ok]]
        aa = a[ok]
        d[ok], _ = project_to_segments(qx, qy, net.ax[aa], net.ay[aa], net.bx[aa], net.by[aa])
    return d, ok


def build_features(points: pd.DataFrame, cands: pd.DataFrame, net: RoadNetwork) -> pd.DataFrame:
    p = cands["point_idx"].to_numpy()
    a = cands["arc_id"].to_numpy()
    dist = cands["dist_m"].to_numpy()

    f = pd.DataFrame({"point_idx": p, "arc_id": a})
    f["dist_m"] = dist
    f["dist_rank"] = cands["rank"].to_numpy()
    best = pd.Series(dist).groupby(p).transform("min").to_numpy()
    f["dist_minus_best_m"] = dist - best
    f["n_candidates"] = pd.Series(p).map(pd.Series(p).value_counts()).to_numpy()

    hv = points["heading_valid"].to_numpy()[p].astype(bool)
    hd = angle_diff_deg(points["heading_deg"].to_numpy()[p], net.bearing[a])
    f["heading_diff_deg"] = np.where(hv, hd, NEUTRAL_HEADING_DIFF)
    f["heading_valid"] = hv.astype(np.int8)

    f["speed_ms"] = points["speed_ms"].to_numpy()[p]
    f["turn_rate_dps"] = points["turn_rate_dps"].to_numpy()[p]
    f["arc_length_m"] = net.length[a]
    t = cands["t_along"].to_numpy()
    f["off_segment"] = ((t <= 0.0) | (t >= 1.0)).astype(np.int8)

    f["prev_dist_m"], _ = _neighbour_dist(points, p, a, -1, net, dist)
    f["next_dist_m"], _ = _neighbour_dist(points, p, a, +1, net, dist)
    acc = np.zeros(len(p))
    cnt = np.zeros(len(p))
    for off in range(-config.CONTEXT_HALF_WINDOW, config.CONTEXT_HALF_WINDOW + 1):
        if off == 0:
            continue
        d, ok = _neighbour_dist(points, p, a, off, net, dist)
        acc += np.where(ok, d, 0.0)
        cnt += ok
    f["context_mean_dist_m"] = np.where(cnt > 0, acc / np.maximum(cnt, 1), dist)

    f["end_out_degree"] = net.out_degree[net.v[a]]
    f["dt_prev_s"] = np.nan_to_num(points["dt_prev_s"].to_numpy()[p], nan=0.0)
    f["fallback"] = cands["fallback"].to_numpy()
    return f


def attach_labels(feats: pd.DataFrame, true_arc: np.ndarray, label_valid: np.ndarray) -> pd.DataFrame:
    p = feats["point_idx"].to_numpy()
    feats["label"] = (feats["arc_id"].to_numpy() == true_arc[p]).astype(np.int8)
    feats["label_valid"] = label_valid[p].astype(np.int8)
    return feats
