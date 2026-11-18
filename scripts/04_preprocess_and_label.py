"""Clean every trip's GPS track and align the reference route to observations.

Outputs
  data/processed/points_<partition>.csv.gz        one row per cleaned observation
  results/preprocessing/trip_log_<partition>.csv  cleaning + label-quality log per trip
  results/preprocessing/summary.csv               totals per partition
  results/figures/preprocessing/*.png

Usage:  python scripts/04_preprocess_and_label.py [--partitions train student_test]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.io import iter_trips
from pathsnap.pipeline import prepare_trip

OUT = config.RESULTS_DIR / "preprocessing"
FIG = config.FIGURES_DIR / "preprocessing"


def run(partitions):
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_logs, all_points = [], []
    for part in partitions:
        logs, pts = [], []
        t0 = time.time()
        for trip in iter_trips(part):
            pt = prepare_trip(trip)
            p = pt.points.copy()
            p.insert(0, "trip_id", trip.trip_id)
            pts.append(p)
            logs.append(pt.log)
        log_df = pd.DataFrame(logs)
        pts_df = pd.concat(pts, ignore_index=True)
        log_df.to_csv(OUT / f"trip_log_{part}.csv", index=False)
        pts_df.to_csv(config.PROCESSED_DIR / f"points_{part}.csv.gz", index=False, float_format="%.6f")
        print(f"[{part}] {len(log_df)} trips, {len(pts_df)} observations, {time.time() - t0:.1f}s")
        all_logs.append(log_df)
        all_points.append(pts_df.assign(partition=part))

    logs = pd.concat(all_logs, ignore_index=True)
    points = pd.concat(all_points, ignore_index=True)

    summary = logs.groupby("partition").agg(
        trips=("trip_id", "count"),
        raw_obs=("raw_obs", "sum"),
        dropped_invalid=("dropped_invalid", "sum"),
        dropped_duplicate_t=("dropped_duplicate_t", "sum"),
        dropped_spikes=("dropped_spikes", "sum"),
        clean_obs=("clean_obs", "sum"),
        total_gaps=("n_gaps", "sum"),
        trips_not_sorted=("was_sorted", lambda s: int((~s.astype(bool)).sum())),
        median_align_dist_m=("align_dist_median_m", "median"),
        mean_route_continuity=("route_continuity", "mean"),
    )
    valid = points.groupby("partition")["label_valid"].agg(["sum", "mean"])
    summary["obs_label_valid"] = valid["sum"]
    summary["obs_label_valid_frac"] = valid["mean"].round(4)
    summary.to_csv(OUT / ("summary.csv" if set(partitions) == set(config.PARTITIONS) else f"summary_{'_'.join(partitions)}.csv"))
    print(summary.T.to_string())

    # --- figures (the histogram is train-only, so skip it when train was not processed)
    tr = points[points["partition"] == "train"]
    if len(tr) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(tr["align_dist_m"], 0, 150), bins=150, color="#3b6ea5")
    ax.axvline(config.LABEL_MAX_ALIGN_DIST_M, color="crimson", ls="--",
               label=f"label_valid threshold = {config.LABEL_MAX_ALIGN_DIST_M:.0f} m")
    ax.set_yscale("log")
    ax.set_xlabel("distance from GPS fix to aligned reference arc (m, clipped at 150)")
    ax.set_ylabel("observations (log)")
    ax.set_title("Train: GPS-to-reference-route distance after monotone alignment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "align_distance_hist_train.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    lt = logs.sort_values("label_valid_frac")
    colors = lt["partition"].map({"train": "#3b6ea5", "student_test": "#e08a2c"})
    ax.bar(range(len(lt)), lt["label_valid_frac"], color=colors)
    ax.set_xticks(range(len(lt)))
    ax.set_xticklabels(lt["trip_id"].str[-3:], rotation=90, fontsize=6)
    ax.set_ylabel("fraction of observations with valid label")
    ax.set_title("Label validity per trip (blue = train, orange = student_test)")
    fig.tight_layout()
    fig.savefig(FIG / "label_valid_frac_per_trip.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", nargs="+", default=list(config.PARTITIONS))
    run(ap.parse_args().partitions)
