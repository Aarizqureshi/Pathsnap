"""Generate candidate arcs per observation and build the feature table.

Steps
  1. (train only) candidate-recall sweep over radius x K -> justifies config values
  2. for each partition: candidates + features + labels for every trip

Outputs
  data/processed/features_<partition>.csv.gz
  results/features/candidate_recall.csv            (train sweep)
  results/features/candidate_stats_<partition>.csv (miss rate etc. per trip)
  results/features/summary.csv
  results/figures/features/candidate_recall.png

Usage:  python scripts/05_build_features.py [--partitions train student_test]
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
from pathsnap.candidates import candidate_recall, generate_candidates
from pathsnap.features import attach_labels, build_features
from pathsnap.io import iter_trips
from pathsnap.pipeline import prepare_trip

OUT = config.RESULTS_DIR / "features"
FIG = config.FIGURES_DIR / "features"
SWEEP_RADII = [10, 15, 20, 30, 40, 50, 75, 100, 150]
SWEEP_K = [1, 2, 4, 8, 12, 16, 24]


def recall_sweep():
    """Recall of the reference arc as a function of radius and K (train only)."""
    max_r = max(SWEEP_RADII)
    rows = []  # per valid observation: distance of true arc and its rank (inf if absent)
    for trip in iter_trips("train"):
        pt = prepare_trip(trip)
        x, y = pt.xy
        pi, ai, d, _ = pt.net.query_radius(x, y, max_r)
        df = pd.DataFrame({"p": pi, "a": ai, "d": d}).sort_values(["p", "d", "a"], kind="stable")
        df["rank"] = df.groupby("p").cumcount()
        true_arc = pt.points["true_arc"].to_numpy()
        hit = df[df["a"].to_numpy() == true_arc[df["p"].to_numpy()]]
        true_d = np.full(len(x), np.inf)
        true_r = np.full(len(x), np.inf)
        true_d[hit["p"].to_numpy()] = hit["d"].to_numpy()
        true_r[hit["p"].to_numpy()] = hit["rank"].to_numpy()
        v = pt.points["label_valid"].to_numpy().astype(bool)
        rows.append(pd.DataFrame({"d": true_d[v], "r": true_r[v]}))
    tr = pd.concat(rows, ignore_index=True)
    res = [{"radius_m": r, "max_k": k, "recall": float(((tr["d"] <= r) & (tr["r"] < k)).mean())}
           for r in SWEEP_RADII for k in SWEEP_K]
    res = pd.DataFrame(res)
    res.to_csv(OUT / "candidate_recall.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in SWEEP_K:
        s = res[res["max_k"] == k]
        ax.plot(s["radius_m"], s["recall"], marker="o", label=f"K = {k}")
    ax.axvline(config.CANDIDATE_RADIUS_M, color="grey", ls=":")
    ax.set_xlabel("candidate radius (m)")
    ax.set_ylabel("recall of reference arc (valid train obs)")
    ax.set_title("Train: candidate-set recall vs radius and K")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "candidate_recall.png", dpi=130)
    plt.close(fig)
    print(res.pivot(index="radius_m", columns="max_k", values="recall").round(4).to_string())


def build(part):
    t0 = time.time()
    feats, stats = [], []
    for trip in iter_trips(part):
        pt = prepare_trip(trip)
        x, y = pt.xy
        cands = generate_candidates(x, y, pt.net)
        f = build_features(pt.points, cands, pt.net)
        f = attach_labels(f, pt.points["true_arc"].to_numpy(), pt.points["label_valid"].to_numpy())
        f.insert(0, "trip_id", trip.trip_id)
        f.insert(2, "obs_idx", pt.points["obs_idx"].to_numpy()[f["point_idx"].to_numpy()])
        feats.append(f)
        st = candidate_recall(cands, pt.points["true_arc"].to_numpy(),
                              pt.points["label_valid"].to_numpy().astype(bool))
        stats.append({"trip_id": trip.trip_id, "n_obs": len(pt.points),
                      "n_candidate_rows": len(cands),
                      "fallback_obs": int(cands.loc[cands["fallback"] == 1, "point_idx"].nunique()), **st})
    feats = pd.concat(feats, ignore_index=True)
    stats = pd.DataFrame(stats)
    feats.to_csv(config.PROCESSED_DIR / f"features_{part}.csv.gz", index=False, float_format="%.4f")
    stats.to_csv(OUT / f"candidate_stats_{part}.csv", index=False)
    v = feats[feats["label_valid"] == 1]
    summ = {
        "partition": part,
        "trips": len(stats),
        "observations": int(stats["n_obs"].sum()),
        "candidate_rows": len(feats),
        "mean_candidates_per_obs": round(len(feats) / stats["n_obs"].sum(), 3),
        "valid_obs": int(stats["n_valid_obs"].sum()),
        "positive_rows": int(v["label"].sum()),
        "negative_rows": int((v["label"] == 0).sum()),
        "candidate_miss_rate": round(1 - v["label"].sum() / stats["n_valid_obs"].sum(), 5),
        "obs_using_fallback": int(stats["fallback_obs"].sum()),
        "seconds": round(time.time() - t0, 1),
    }
    print(summ)
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", nargs="+", default=list(config.PARTITIONS))
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_sweep:
        recall_sweep()
    pd.DataFrame([build(p) for p in args.partitions]).to_csv(OUT / "summary.csv", index=False)
