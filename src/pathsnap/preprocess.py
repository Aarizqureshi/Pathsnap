"""GPS track cleaning and per-observation kinematics.

Steps (each one is counted in the returned log so it can be reported):

1. drop non-finite or out-of-range coordinates
2. sort by timestamp (stable) and drop duplicated timestamps
3. project to the trip's local metric plane
4. remove isolated "spike" fixes whose implied speed to BOTH neighbours is
   implausible while the neighbours themselves are consistent
5. split the track into continuous segments at time gaps > GAP_THRESHOLD_S
6. compute speed, heading and turn rate per observation (within a segment)

Nothing here is learned from labels, so applying it identically to every
partition cannot leak information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .geo import LocalProjection, angle_diff_deg, bearing_deg


def _remove_spikes(x, y, t, max_speed):
    """Indices to keep after dropping single-point GPS jumps (iterated)."""
    keep = np.ones(len(x), dtype=bool)
    for _ in range(3):
        idx = np.flatnonzero(keep)
        if len(idx) < 3:
            break
        xi, yi, ti = x[idx], y[idx], t[idx]
        dt = np.diff(ti)
        v = np.hypot(np.diff(xi), np.diff(yi)) / np.maximum(dt, 1e-9)
        v_in = np.r_[0.0, v]
        v_out = np.r_[v, 0.0]
        # speed if the middle point were skipped
        dt2 = ti[2:] - ti[:-2]
        v_skip = np.r_[0.0, np.hypot(xi[2:] - xi[:-2], yi[2:] - yi[:-2]) / np.maximum(dt2, 1e-9), 0.0]
        spike = (v_in > max_speed) & (v_out > max_speed) & (v_skip <= max_speed)
        if not spike.any():
            break
        keep[idx[spike]] = False
    return keep


def clean_track(track: pd.DataFrame, proj: LocalProjection | None = None):
    """Return ``(points, log)``.

    ``points`` has one row per retained observation with columns
    ``obs_idx`` (row in the raw .track file), ``t, lon, lat, x, y,
    dt_prev_s, segment_id, speed_ms, heading_deg, heading_valid, turn_rate_dps``.
    """
    log = {"raw_obs": len(track)}
    df = track.copy()
    df["obs_idx"] = np.arange(len(df))

    finite = np.isfinite(df[["lon", "lat", "t"]].to_numpy()).all(axis=1)
    in_range = (df["lon"].between(*config.LON_RANGE) & df["lat"].between(*config.LAT_RANGE)).to_numpy()
    log["dropped_invalid"] = int((~(finite & in_range)).sum())
    df = df[finite & in_range]

    log["was_sorted"] = bool(df["t"].is_monotonic_increasing)
    df = df.sort_values("t", kind="stable")
    dup = df["t"].duplicated(keep="first").to_numpy()
    log["dropped_duplicate_t"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)

    if proj is None:
        proj = LocalProjection.from_points(df["lon"], df["lat"])
    x, y = proj.to_xy(df["lon"].to_numpy(), df["lat"].to_numpy())
    t = df["t"].to_numpy(float)

    keep = _remove_spikes(x, y, t, config.MAX_PLAUSIBLE_SPEED_MS)
    log["dropped_spikes"] = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    x, y, t = x[keep], y[keep], t[keep]
    df["x"], df["y"] = x, y

    dt = np.r_[np.nan, np.diff(t)]
    gap = np.r_[True, dt[1:] > config.GAP_THRESHOLD_S]
    seg = np.cumsum(gap) - 1
    df["dt_prev_s"] = dt
    df["segment_id"] = seg
    log["n_segments"] = int(seg.max() + 1) if len(seg) else 0
    log["n_gaps"] = max(log["n_segments"] - 1, 0)
    log["max_gap_s"] = float(np.nanmax(dt)) if len(dt) > 1 else 0.0

    # ---- kinematics, computed inside each segment only
    n = len(df)
    w = config.HEADING_HALF_WINDOW
    i = np.arange(n)
    lo = np.maximum(i - w, 0)
    hi = np.minimum(i + w, n - 1)
    # pull window ends back so they stay inside the same segment
    seg_start = np.r_[0, np.flatnonzero(np.diff(seg)) + 1]
    seg_end = np.r_[seg_start[1:] - 1, n - 1]
    lo = np.maximum(lo, seg_start[seg])
    hi = np.minimum(hi, seg_end[seg])

    ddx = x[hi] - x[lo]
    ddy = y[hi] - y[lo]
    disp = np.hypot(ddx, ddy)
    span = t[hi] - t[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.where(span > 0, disp / span, 0.0)
    heading = bearing_deg(ddx, ddy)
    heading_valid = disp >= config.MIN_DISPLACEMENT_FOR_HEADING_M

    # turn rate: heading change between the incoming and outgoing halves
    mid = i
    h_in = bearing_deg(x[mid] - x[lo], y[mid] - y[lo])
    h_out = bearing_deg(x[hi] - x[mid], y[hi] - y[mid])
    span_turn = np.maximum(t[hi] - t[lo], 1.0)
    turn = angle_diff_deg(h_in, h_out) / span_turn
    ok_turn = (np.hypot(x[mid] - x[lo], y[mid] - y[lo]) >= 1.0) & (np.hypot(x[hi] - x[mid], y[hi] - y[mid]) >= 1.0)
    turn = np.where(ok_turn, turn, 0.0)

    df["speed_ms"] = speed
    df["heading_deg"] = heading
    df["heading_valid"] = heading_valid.astype(np.int8)
    df["turn_rate_dps"] = turn
    log["clean_obs"] = n
    log["heading_valid_frac"] = float(heading_valid.mean()) if n else np.nan

    cols = ["obs_idx", "t", "lon", "lat", "x", "y", "dt_prev_s", "segment_id",
            "speed_ms", "heading_deg", "heading_valid", "turn_rate_dps"]
    return df[cols], log
