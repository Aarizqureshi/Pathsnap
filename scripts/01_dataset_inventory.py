"""Dataset inventory and raw data-quality checks (no cleaning applied).

For every trip in every partition this records file availability/sizes,
observation counts, timestamp behaviour, geographic extent, road-network
size and connectivity, and reference-route validity.

Outputs
  results/eda/inventory.csv            one row per trip
  results/eda/partition_summary.csv    totals / medians per partition
  results/eda/missing_values.csv       NaN / invalid / duplicate / ordering checks per partition
  results/eda/projection_check.csv     local projection vs haversine distances

Usage:  python scripts/01_dataset_inventory.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.geo import LocalProjection, haversine_m
from pathsnap.io import FILE_TYPES, iter_trips
from pathsnap.network import RoadNetwork, route_continuity

OUT = config.RESULTS_DIR / "eda"


def trip_row(trip):
    tr = trip.track
    row = {"partition": trip.partition, "trip_id": trip.trip_id}
    for ft in FILE_TYPES:
        f = trip.files[ft]
        row[f"has_{ft}"] = f.exists()
        row[f"{ft}_kb"] = round(f.stat().st_size / 1024, 1) if f.exists() else 0.0

    vals = tr[["lon", "lat", "t"]].to_numpy()
    row["n_obs"] = len(tr)
    row["nan_values"] = int(np.isnan(vals).sum())
    row["out_of_range_coords"] = int((~tr["lon"].between(-180, 180) | ~tr["lat"].between(-90, 90)).sum())
    row["duplicate_timestamps"] = int(tr["t"].duplicated().sum())
    dt = np.diff(tr["t"].to_numpy())
    row["non_increasing_steps"] = int((dt <= 0).sum())
    row["duration_s"] = float(tr["t"].iloc[-1] - tr["t"].iloc[0])
    row["dt_median_s"] = float(np.median(dt))
    row["frac_dt_eq_1"] = float(np.mean(dt == 1))
    row["n_gaps_gt_threshold"] = int((dt > config.GAP_THRESHOLD_S).sum())
    row["max_gap_s"] = float(dt.max())
    row["lon_center"] = float(tr["lon"].median())
    row["lat_center"] = float(tr["lat"].median())

    proj = LocalProjection.from_points(tr["lon"], tr["lat"])
    x, y = proj.to_xy(tr["lon"].to_numpy(), tr["lat"].to_numpy())
    row["extent_ew_km"] = float((x.max() - x.min()) / 1000)
    row["extent_ns_km"] = float((y.max() - y.min()) / 1000)
    step = np.hypot(np.diff(x), np.diff(y))
    row["path_length_km"] = float(step.sum() / 1000)
    speed = step / np.maximum(dt, 1e-9)
    row["speed_median_ms"] = float(np.median(speed))
    row["speed_p99_ms"] = float(np.percentile(speed, 99))
    row["steps_over_max_speed"] = int((speed > config.MAX_PLAUSIBLE_SPEED_MS).sum())

    hv = haversine_m(tr["lon"].to_numpy()[:-1], tr["lat"].to_numpy()[:-1],
                     tr["lon"].to_numpy()[1:], tr["lat"].to_numpy()[1:])
    row["proj_max_abs_err_m"] = float(np.max(np.abs(step - hv))) if len(hv) else 0.0

    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    row["arc_ids_valid"] = bool(len(trip.arcs) == 0 or trip.arcs.to_numpy().max() < len(trip.nodes))
    row.update(net.connectivity_summary())
    if trip.route is not None:
        r = trip.route
        row["route_len_arcs"] = len(r)
        row["route_ids_valid"] = bool(r.max() < net.n_arcs and r.min() >= 0)
        row["route_continuity"] = route_continuity(r, net)
        row["route_length_km"] = float(net.length[r].sum() / 1000)
        row["route_repeated_arcs"] = int(len(r) - len(np.unique(r)))
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for part in config.PARTITIONS:
        for trip in iter_trips(part):
            rows.append(trip_row(trip))
            print(f"{part}/{trip.trip_id}: {rows[-1]['n_obs']} obs, {rows[-1]['n_arcs']} arcs")
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "inventory.csv", index=False, float_format="%.4f")

    g = inv.groupby("partition")
    summary = pd.DataFrame({
        "trips": g.size(),
        "total_obs": g["n_obs"].sum(),
        "median_obs_per_trip": g["n_obs"].median(),
        "min_obs_per_trip": g["n_obs"].min(),
        "max_obs_per_trip": g["n_obs"].max(),
        "median_duration_min": g["duration_s"].median() / 60,
        "median_path_km": g["path_length_km"].median(),
        "median_arcs": g["n_arcs"].median(),
        "max_arcs": g["n_arcs"].max(),
        "trips_with_gaps": g["n_gaps_gt_threshold"].apply(lambda s: int((s > 0).sum())),
        "trips_multi_component_network": g["n_components"].apply(lambda s: int((s > 1).sum())),
        "median_route_arcs": g["route_len_arcs"].median(),
        "trips_route_not_fully_connected": g["route_continuity"].apply(lambda s: int((s < 1).sum())),
    })
    summary.round(3).to_csv(OUT / "partition_summary.csv")
    print(summary.round(2).T.to_string())

    miss_cols = ["nan_values", "out_of_range_coords", "duplicate_timestamps", "non_increasing_steps",
                 "steps_over_max_speed", "isolated_nodes", "self_loops", "duplicate_arcs", "zero_length_arcs"]
    missing = g[miss_cols].sum()
    missing["files_missing"] = g[[f"has_{f}" for f in FILE_TYPES]].apply(lambda d: int((~d).sum().sum()))
    missing.to_csv(OUT / "missing_values.csv")
    print(missing.T.to_string())

    inv[["partition", "trip_id", "extent_ew_km", "extent_ns_km", "proj_max_abs_err_m"]].to_csv(
        OUT / "projection_check.csv", index=False, float_format="%.6f")
    print(f"max |projected - haversine| over all consecutive fixes: {inv['proj_max_abs_err_m'].max():.4f} m")


if __name__ == "__main__":
    main()
