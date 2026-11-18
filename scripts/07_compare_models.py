"""Train and compare the five candidate classifiers.

Protocol
  * CV: GroupKFold (groups = trip) on a fixed CV_OBS_FRAC sample of valid
    train observations; every model sees the same rows and folds.
  * Final fit: all valid train rows (KNN / SVM capped, see config).
  * Student-test: every metric on valid observations; route continuity on
    the classifier-only top-1 sequence of every observation.
  * Selection rule: highest mean CV top-1 accuracy (tie: lower CV log loss).
    Student-test results are reported next to it as the development check.

Outputs
  results/models/cv_folds.csv, comparison.csv, model_configs.json,
  selected_model.json, results/figures/models/*.png,
  models/<name>.joblib (not committed), data/processed/proba_student_test_<name>.csv.gz

Usage:  python scripts/07_compare_models.py [--models logreg gnb knn rf svm_rbf]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap import config
from pathsnap.io import load_trip
from pathsnap.metrics import (candidate_metrics, candidate_miss_rate, confusion, pick_top1,
                              route_continuity, top1_accuracy)
from pathsnap.models import SPECS, fit_model, load_features, predict_proba, sample_observations, valid

OUT = config.RESULTS_DIR / "models"
FIG = config.FIGURES_DIR / "models"


def cross_validate(name, cv_df, folds):
    rows = []
    groups = cv_df["trip_id"].to_numpy()
    for k, (tr_idx, va_idx) in enumerate(folds.split(cv_df, groups=groups)):
        tr, va = cv_df.iloc[tr_idx], cv_df.iloc[va_idx]
        t0 = time.perf_counter()
        model, n_rows = fit_model(name, tr)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = predict_proba(model, va)
        pred_s = time.perf_counter() - t0
        m = candidate_metrics(va["label"].to_numpy(), p)
        m.update(model=name, fold=k, train_rows=n_rows, val_rows=len(va),
                 top1=top1_accuracy(va, p), fit_s=fit_s, predict_s=pred_s)
        rows.append(m)
        print(f"  {name} fold {k}: top1={m['top1']:.4f} f1={m['f1_macro']:.4f} "
              f"logloss={m['log_loss']:.4f} fit={fit_s:.1f}s")
    return rows


def continuity_on(test_all, score, trips):
    """Classifier-only top-1 for every observation, in time order, per trip.

    Returns pooled continuity (strict: next arc starts where the previous ends)
    and pooled continuity allowing up to 3 arcs per step.
    """
    best = pick_top1(test_all, score).sort_values(["trip_id", "point_idx"])
    changes, jumps1, jumps3 = 0, 0, 0
    for tid, g in best.groupby("trip_id"):
        u, v = trips[tid]
        _, n, j1 = route_continuity(g["arc_id"].to_numpy(), u, v, max_hops=1)
        _, _, j3 = route_continuity(g["arc_id"].to_numpy(), u, v, max_hops=3)
        changes += n
        jumps1 += j1
        jumps3 += j3
    return 1 - jumps1 / max(changes, 1), 1 - jumps3 / max(changes, 1)


def main(names):
    for d in (OUT, FIG, config.MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    train = valid(load_features("train"))
    test_all = load_features("student_test")
    test = valid(test_all)
    cv_df = sample_observations(train, config.CV_OBS_FRAC).reset_index(drop=True)
    folds = GroupKFold(n_splits=config.CV_FOLDS)
    print(f"train valid rows {len(train)}, CV sample rows {len(cv_df)} "
          f"({cv_df['trip_id'].nunique()} trips), student-test rows {len(test_all)}")

    trips = {}
    for tid in test_all["trip_id"].unique():
        t = load_trip(config.DATA_DIR / "student_test" / tid, "student_test")
        trips[tid] = (t.arcs["u"].to_numpy(), t.arcs["v"].to_numpy())

    train_score_sample = cap_for_train_score(train)
    fold_rows, summary, confs = [], [], {}
    for name in names:
        print(f"\n=== {name} ({SPECS[name].family})")
        fold_rows += cross_validate(name, cv_df, folds)
        cvf = pd.DataFrame([r for r in fold_rows if r["model"] == name])

        t0 = time.perf_counter()
        model, n_rows = fit_model(name, train)
        fit_s = time.perf_counter() - t0
        joblib.dump(model, config.MODELS_DIR / f"{name}.joblib", compress=3)

        t0 = time.perf_counter()
        p_all = predict_proba(model, test_all)
        pred_s = time.perf_counter() - t0
        n_obs = test_all[["trip_id", "point_idx"]].drop_duplicates().shape[0]
        pd.DataFrame({"trip_id": test_all["trip_id"], "point_idx": test_all["point_idx"],
                      "arc_id": test_all["arc_id"], "proba": p_all}).to_csv(
            config.PROCESSED_DIR / f"proba_student_test_{name}.csv.gz", index=False, float_format="%.6f")

        vmask = test_all["label_valid"].to_numpy() == 1
        m = candidate_metrics(test["label"].to_numpy(), p_all[vmask])
        cont_pooled, cont_3hop = continuity_on(test_all, p_all, trips)
        confs[name] = confusion(test["label"].to_numpy(), p_all[vmask])
        tr_s = train_score_sample
        tr_p = predict_proba(model, tr_s)
        tr_m = candidate_metrics(tr_s["label"].to_numpy(), tr_p)
        summary.append({
            "model": name, "family": SPECS[name].family, "config": json.dumps(SPECS[name].defaults),
            "notes": SPECS[name].notes, "final_train_rows": n_rows,
            "cv_f1_macro_mean": cvf["f1_macro"].mean(), "cv_f1_macro_std": cvf["f1_macro"].std(),
            "cv_top1_mean": cvf["top1"].mean(), "cv_top1_std": cvf["top1"].std(),
            "cv_log_loss_mean": cvf["log_loss"].mean(),
            "train_f1_macro": tr_m["f1_macro"], "train_top1": top1_accuracy(tr_s, tr_p),
            "test_accuracy": m["accuracy"], "test_precision_macro": m["precision_macro"],
            "test_recall_macro": m["recall_macro"], "test_f1_macro": m["f1_macro"],
            "test_log_loss": m["log_loss"], "test_top1": top1_accuracy(test_all, p_all),
            "test_candidate_miss_rate": candidate_miss_rate(test_all),
            "test_route_continuity": cont_pooled, "test_route_continuity_3hop": cont_3hop,
            "fit_s": fit_s, "predict_s": pred_s, "predict_ms_per_1k_obs": 1e6 * pred_s / n_obs,
        })
        s = summary[-1]
        print(f"  student-test: top1={s['test_top1']:.4f} f1={s['test_f1_macro']:.4f} "
              f"logloss={s['test_log_loss']:.4f} continuity={s['test_route_continuity']:.4f}")

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(OUT / "cv_folds.csv", index=False, float_format="%.5f")
    comp = pd.DataFrame(summary)
    comp.to_csv(OUT / "comparison.csv", index=False, float_format="%.5f")
    (OUT / "model_configs.json").write_text(json.dumps(
        {n: {"family": SPECS[n].family, "defaults": SPECS[n].defaults, "notes": SPECS[n].notes,
             "max_train_rows": SPECS[n].max_train_rows} for n in names}, indent=2))

    ranked = comp.sort_values(["cv_top1_mean", "cv_log_loss_mean"], ascending=[False, True])
    best = ranked.iloc[0]
    (OUT / "selected_model.json").write_text(json.dumps({
        "selected": best["model"],
        "rule": "highest mean CV top-1 accuracy (GroupKFold by trip), tie-break lower CV log loss",
        "ranking": ranked[["model", "cv_top1_mean", "cv_f1_macro_mean", "cv_log_loss_mean",
                           "test_top1", "test_f1_macro"]].round(5).to_dict("records"),
    }, indent=2))
    print("\n", comp[["model", "cv_f1_macro_mean", "cv_top1_mean", "test_top1", "test_f1_macro",
                      "test_log_loss", "test_route_continuity", "fit_s"]].round(4).to_string(index=False))
    print(f"\nselected: {best['model']}")
    plots(comp, folds_df, confs)


def cap_for_train_score(train):
    """Same sample for every model when measuring training-set fit (overfitting check)."""
    return sample_observations(train, 0.05, seed=config.RANDOM_SEED + 1)


def plots(comp, folds_df, confs):
    names = comp["model"].tolist()
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, n in enumerate(names):
        f = folds_df[folds_df["model"] == n]["f1_macro"]
        ax.scatter(np.full(len(f), i), f, color="#999999", s=14, zorder=3)
    ax.bar(x, comp["cv_f1_macro_mean"], yerr=comp["cv_f1_macro_std"], color="#3b6ea5", capsize=5)
    ax.set_xticks(x, names)
    ax.set_ylabel("macro F1 (5-fold GroupKFold)")
    ax.set_ylim(max(0, comp["cv_f1_macro_mean"].min() - 0.1), 1)
    ax.set_title("Cross-validation macro F1 (bar = mean, dots = folds)")
    fig.tight_layout(); fig.savefig(FIG / "cv_macro_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.38
    ax.bar(x - w / 2, comp["test_top1"], w, label="top-1 accuracy", color="#3b6ea5")
    ax.bar(x + w / 2, comp["test_f1_macro"], w, label="macro F1", color="#e08a2c")
    for i, (a, b) in enumerate(zip(comp["test_top1"], comp["test_f1_macro"])):
        ax.text(i - w / 2, a + 0.005, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.005, f"{b:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylim(max(0, min(comp["test_top1"].min(), comp["test_f1_macro"].min()) - 0.1), 1.02)
    ax.legend()
    ax.set_title("Student-test: observation top-1 accuracy and candidate macro F1")
    fig.tight_layout(); fig.savefig(FIG / "student_test_top1_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, comp["cv_log_loss_mean"], w, label="CV", color="#3b6ea5")
    ax.bar(x + w / 2, comp["test_log_loss"], w, label="student-test", color="#e08a2c")
    ax.set_xticks(x, names)
    ax.set_ylabel("log loss (lower is better)")
    ax.legend()
    ax.set_title("Probability quality: log loss")
    fig.tight_layout(); fig.savefig(FIG / "log_loss.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(x, comp["fit_s"], color="#3b6ea5")
    axes[0].set_title("final fit time (s)")
    axes[1].bar(x, comp["predict_s"], color="#e08a2c")
    axes[1].set_title("student-test inference time (s)")
    for a in axes:
        a.set_xticks(x, names)
    fig.tight_layout(); fig.savefig(FIG / "timing.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        cm = confs[n]
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{cmn[i, j]:.1%}", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], ["other", "ref arc"])
        ax.set_yticks([0, 1], ["other", "ref arc"])
        ax.set_xlabel("predicted")
        ax.set_title(n)
    axes[0].set_ylabel("actual")
    fig.suptitle("Student-test confusion matrices (threshold 0.5, row-normalised %)")
    fig.tight_layout(); fig.savefig(FIG / "confusion_matrices.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(comp["test_route_continuity"], comp["test_top1"], s=60, color="#3b6ea5")
    for _, r in comp.iterrows():
        ax.annotate(r["model"], (r["test_route_continuity"], r["test_top1"]), xytext=(5, 5),
                    textcoords="offset points")
    ax.set_xlabel("route continuity of classifier-only output")
    ax.set_ylabel("top-1 accuracy")
    ax.set_title("Student-test: accuracy vs route continuity (before HMM)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "accuracy_vs_continuity.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(SPECS))
    main(ap.parse_args().models)
