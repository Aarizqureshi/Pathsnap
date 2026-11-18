"""Analyse the engineered features before any model is trained.

Uses TRAIN rows with label_valid == 1 for all statistics (no student_test
labels influence feature design). Student-test numbers are reported only
for the two rule-based reference baselines.

Outputs
  results/features/class_balance.csv
  results/features/feature_summary.csv    mean/median per class + univariate ROC-AUC
  results/features/rule_baselines.csv     nearest-arc and heading-aware nearest-arc top-1
  results/figures/features/feature_distributions.png
  results/figures/features/feature_correlation.png
  results/figures/features/feature_auc.png

Usage:  python scripts/06_feature_analysis.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap import config

OUT = config.RESULTS_DIR / "features"
FIG = config.FIGURES_DIR / "features"


def roc_auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) ROC-AUC, ties get average ranks."""
    ranks = pd.Series(score).rank(method="average").to_numpy()
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def top1_accuracy(df: pd.DataFrame, score_col: str, n_valid_obs: int) -> float:
    """Pick the lowest-score candidate per observation; obs without the true arc count as wrong."""
    best = df.sort_values(["trip_id", "point_idx", score_col, "arc_id"], kind="stable")
    best = best.drop_duplicates(["trip_id", "point_idx"])
    return float(best["label"].sum() / n_valid_obs)


def rule_baselines(df: pd.DataFrame) -> dict:
    v = df[df["label_valid"] == 1].copy()
    n_obs = v.groupby(["trip_id", "point_idx"]).ngroups
    v["nearest"] = v["dist_m"]
    # heading-aware: candidates pointing the wrong way (>90 deg) are pushed back
    wrong_way = (v["heading_valid"] == 1) & (v["heading_diff_deg"] > 90)
    v["heading_nearest"] = v["dist_m"] + 1e6 * wrong_way
    return {"nearest_arc_top1": top1_accuracy(v, "nearest", n_obs),
            "heading_aware_nearest_top1": top1_accuracy(v, "heading_nearest", n_obs),
            "valid_obs": n_obs}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(config.PROCESSED_DIR / "features_train.csv.gz")
    st = pd.read_csv(config.PROCESSED_DIR / "features_student_test.csv.gz")

    trv = tr[tr["label_valid"] == 1]
    y = trv["label"].to_numpy()
    balance = pd.DataFrame({
        "partition": ["train", "student_test"],
        "rows_valid": [len(trv), int((st["label_valid"] == 1).sum())],
        "positives": [int(y.sum()), int(st.loc[st["label_valid"] == 1, "label"].sum())],
    })
    balance["negatives"] = balance["rows_valid"] - balance["positives"]
    balance["neg_per_pos"] = (balance["negatives"] / balance["positives"]).round(2)
    balance.to_csv(OUT / "class_balance.csv", index=False)
    print(balance.to_string(index=False))

    rows = []
    for c in config.FEATURE_COLUMNS:
        x = trv[c].to_numpy(float)
        auc = roc_auc(x, y)
        rows.append({
            "feature": c,
            "mean_pos": x[y == 1].mean(), "mean_neg": x[y == 0].mean(),
            "median_pos": np.median(x[y == 1]), "median_neg": np.median(x[y == 0]),
            "roc_auc": auc,
            "separability": abs(auc - 0.5) * 2,  # 0 = useless, 1 = perfect, direction-free
        })
    fs = pd.DataFrame(rows).sort_values("separability", ascending=False)
    fs.round(4).to_csv(OUT / "feature_summary.csv", index=False)
    print(fs.round(3).to_string(index=False))

    rb = pd.DataFrame([{"partition": "train", **rule_baselines(tr)},
                       {"partition": "student_test", **rule_baselines(st)}])
    rb.round(4).to_csv(OUT / "rule_baselines.csv", index=False)
    print(rb.round(4).to_string(index=False))

    # --- figures
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(fs["feature"][::-1], fs["separability"][::-1], color="#3b6ea5")
    ax.set_xlabel("univariate separability  |2 x ROC-AUC - 1|  (train, valid rows)")
    ax.set_title("How well each feature alone separates true vs other candidates")
    fig.tight_layout()
    fig.savefig(FIG / "feature_auc.png", dpi=130)
    plt.close(fig)

    show = ["dist_m", "dist_minus_best_m", "heading_diff_deg", "context_mean_dist_m",
            "prev_dist_m", "dist_rank", "speed_ms", "arc_length_m", "n_candidates"]
    clip = {"dist_m": 50, "dist_minus_best_m": 50, "context_mean_dist_m": 80, "prev_dist_m": 80,
            "speed_ms": 40, "arc_length_m": 400}
    sample = trv.sample(n=min(300_000, len(trv)), random_state=config.RANDOM_SEED)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, c in zip(axes.ravel(), show):
        v = sample[c].clip(upper=clip.get(c, None))
        bins = 40 if c not in ("dist_rank", "n_candidates") else np.arange(-0.5, 13.5, 1)
        ax.hist(v[sample["label"] == 0], bins=bins, density=True, alpha=0.55, label="other candidate", color="#999999")
        ax.hist(v[sample["label"] == 1], bins=bins, density=True, alpha=0.65, label="reference arc", color="#d0452c")
        ax.set_title(c, fontsize=10)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Train feature distributions: reference arc vs other candidates (density)")
    fig.tight_layout()
    fig.savefig(FIG / "feature_distributions.png", dpi=130)
    plt.close(fig)

    corr = sample[config.FEATURE_COLUMNS].corr(method="spearman")
    corr.round(3).to_csv(OUT / "feature_correlation_spearman.csv")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Spearman correlation between features (train sample)")
    fig.tight_layout()
    fig.savefig(FIG / "feature_correlation.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
