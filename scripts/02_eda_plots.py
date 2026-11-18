"""Exploratory plots of the raw dataset.

Reads results/eda/inventory.csv (run 01_dataset_inventory.py first) plus the
raw tracks. Figures go to results/figures/eda/.

Usage:  python scripts/02_eda_plots.py [--example-trip 00000005]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from pathsnap import config
from pathsnap.geo import LocalProjection
from pathsnap.io import iter_trips, load_trip
from pathsnap.network import RoadNetwork

FIG = config.FIGURES_DIR / "eda"
PART_COLORS = {"train": "#3b6ea5", "student_test": "#e08a2c"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)
    print("saved", FIG / name)


def main(example_trip, example_partition):
    FIG.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(config.RESULTS_DIR / "eda" / "inventory.csv", dtype={"trip_id": str})

    # 1. where are the trips?
    fig, ax = plt.subplots(figsize=(10, 5))
    for part, d in inv.groupby("partition"):
        ax.scatter(d["lon_center"], d["lat_center"], s=25, label=f"{part} ({len(d)})", color=PART_COLORS[part])
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Trip locations (median GPS position) - trips span 5 continents")
    save(fig, "trip_locations.png")

    # 2. observations per trip / duration / path length
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, lab in zip(axes, ["n_obs", "duration_s", "path_length_km"],
                            ["GPS observations per trip", "duration (s)", "path length (km)"]):
        for part, d in inv.groupby("partition"):
            ax.hist(d[col], bins=np.logspace(np.log10(max(inv[col].min(), 1)), np.log10(inv[col].max()), 25),
                    alpha=0.7, label=part, color=PART_COLORS[part])
        ax.set_xscale("log")
        ax.set_xlabel(lab)
        ax.set_ylabel("trips")
    axes[0].legend()
    fig.suptitle("Trajectory size distributions")
    save(fig, "trajectory_sizes.png")

    # 3. sampling intervals and speeds (raw, all train + student_test)
    dts, speeds = [], []
    for part in config.PARTITIONS:
        for trip in iter_trips(part):
            t = trip.track["t"].to_numpy()
            proj = LocalProjection.from_points(trip.track["lon"], trip.track["lat"])
            x, y = proj.to_xy(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
            dt = np.diff(t)
            dts.append(dt)
            speeds.append(np.hypot(np.diff(x), np.diff(y))[dt == 1])
    dts = np.concatenate(dts)
    speeds = np.concatenate(speeds)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    vals, counts = np.unique(np.clip(dts, 0, 31), return_counts=True)
    axes[0].bar(vals, counts, color="#3b6ea5")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("time between consecutive fixes (s, values > 30 pooled at 31)")
    axes[0].set_ylabel("count (log)")
    axes[0].set_title(f"Sampling intervals: {np.mean(dts == 1):.1%} exactly 1 s")
    axes[1].hist(np.clip(speeds, 0, 50), bins=100, color="#3b6ea5")
    axes[1].set_xlabel("speed between 1-s fixes (m/s, clipped at 50)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Implied speed distribution")
    save(fig, "sampling_and_speed.png")

    # 4. gaps per trip
    fig, ax = plt.subplots(figsize=(11, 4))
    d = inv.sort_values("max_gap_s")
    ax.bar(range(len(d)), d["max_gap_s"], color=d["partition"].map(PART_COLORS))
    ax.set_yscale("log")
    ax.axhline(config.GAP_THRESHOLD_S, color="crimson", ls="--", label=f"gap threshold {config.GAP_THRESHOLD_S:.0f} s")
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["trip_id"].str[-3:], rotation=90, fontsize=6)
    ax.set_ylabel("largest time gap (s, log)")
    ax.set_title("Largest GPS gap per trip (blue = train, orange = student_test)")
    ax.legend()
    save(fig, "max_gap_per_trip.png")

    # 5. road network size and connectivity
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for part, d in inv.groupby("partition"):
        axes[0].scatter(d["n_nodes"], d["n_arcs"], s=20, label=part, color=PART_COLORS[part])
        axes[1].scatter(d["n_arcs"], d["n_components"], s=20, label=part, color=PART_COLORS[part])
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("nodes"); axes[0].set_ylabel("arcs (directed)")
    axes[0].set_title("Road-network size per trip")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("arcs"); axes[1].set_ylabel("weakly connected components")
    axes[1].set_title("Network fragmentation")
    axes[0].legend()
    save(fig, "network_size_connectivity.png")

    # 6. example trajectory over its road network
    trip = load_trip(config.DATA_DIR / example_partition / example_trip, example_partition)
    proj = LocalProjection.from_points(trip.track["lon"], trip.track["lat"])
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    x, y = proj.to_xy(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
    pad = 300
    box = (x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad)
    inside = ((np.maximum(net.ax, net.bx) >= box[0]) & (np.minimum(net.ax, net.bx) <= box[1]) &
              (np.maximum(net.ay, net.by) >= box[2]) & (np.minimum(net.ay, net.by) <= box[3]))
    fig, ax = plt.subplots(figsize=(9, 9))
    segs = np.stack([np.c_[net.ax[inside], net.ay[inside]], np.c_[net.bx[inside], net.by[inside]]], axis=1)
    ax.add_collection(LineCollection(segs, colors="#bbbbbb", linewidths=0.6, label="road network"))
    if trip.route is not None:
        r = trip.route
        rs = np.stack([np.c_[net.ax[r], net.ay[r]], np.c_[net.bx[r], net.by[r]]], axis=1)
        ax.add_collection(LineCollection(rs, colors="#2a9d3a", linewidths=5, alpha=0.5, label="reference route"))
    ax.scatter(x, y, s=2, color="#1f4fd1", alpha=0.5, label="GPS fixes", zorder=3)
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal")
    ax.set_xlabel("east (m, local projection)"); ax.set_ylabel("north (m)")
    ax.legend(loc="upper right")
    ax.set_title(f"Trip {trip.trip_id} ({example_partition}): GPS vs reference route")
    save(fig, f"example_trip_{trip.trip_id}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--example-trip", default="00000046")
    ap.add_argument("--example-partition", default="train")
    a = ap.parse_args()
    main(a.example_trip, a.example_partition)
