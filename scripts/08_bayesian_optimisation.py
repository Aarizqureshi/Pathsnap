"""Bayesian hyperparameter optimisation of the selected model (Optuna TPE).

* Model: read from results/models/selected_model.json (override with --model).
* Objective: mean observation-level top-1 accuracy over TUNE_FOLDS-fold
  GroupKFold (groups = trip) on a TUNE_OBS_FRAC sample of valid train
  observations. Macro F1 and log loss are recorded per trial as well.
* Sampler: Tree-structured Parzen Estimator, seed = RANDOM_SEED.
* Only train data is used for the search; student-test is used once at the
  end to compare the untuned and tuned models on identical metrics.

Outputs
  results/tuning/trials.csv, best_params.json, tuned_vs_untuned.csv, search_space.json
  results/figures/tuning/*.png
  models/final_model.joblib + models/model_card.json
  data/processed/proba_student_test_final.csv.gz

Usage:  python scripts/08_bayesian_optimisation.py [--model rf] [--trials 25]
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
import optuna
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold

from pathsnap import config
from pathsnap.metrics import candidate_metrics, top1_accuracy
from pathsnap.models import FEATURES, SPECS, fit_model, load_features, predict_proba, sample_observations, valid

OUT = config.RESULTS_DIR / "tuning"
FIG = config.FIGURES_DIR / "tuning"

# Ranges are centred on each model's default and bounded by laptop compute and model size.
SEARCH_SPACES = {
    "rf": {
        "n_estimators": ("int", 50, 200, False),
        "max_leaf_nodes": ("int", 256, 4096, True),   # upper bound keeps final_model.joblib well under 100 MB
        "min_samples_leaf": ("int", 5, 100, True),
        "max_features": ("cat", ["sqrt", "log2", 0.5]),
    },
    "logreg": {"C": ("float", 1e-3, 1e2, True)},
    "gnb": {"var_smoothing": ("float", 1e-12, 1e-3, True)},
    "knn": {"n_neighbors": ("int", 5, 200, True), "weights": ("cat", ["uniform", "distance"])},
    "svm_rbf": {"C": ("float", 1e-2, 1e2, True), "gamma": ("float", 1e-3, 1.0, True)},
}


def suggest(trial, space):
    p = {}
    for k, spec in space.items():
        kind = spec[0]
        if kind == "int":
            p[k] = trial.suggest_int(k, spec[1], spec[2], log=spec[3])
        elif kind == "float":
            p[k] = trial.suggest_float(k, spec[1], spec[2], log=spec[3])
        else:
            p[k] = trial.suggest_categorical(k, spec[1])
    return p


def cv_score(name, params, df, n_folds):
    res = []
    for tr_idx, va_idx in GroupKFold(n_splits=n_folds).split(df, groups=df["trip_id"].to_numpy()):
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        model, _ = fit_model(name, tr, params)
        p = predict_proba(model, va)
        m = candidate_metrics(va["label"].to_numpy(), p)
        res.append({"top1": top1_accuracy(va, p), "f1_macro": m["f1_macro"], "log_loss": m["log_loss"]})
    return pd.DataFrame(res).mean().to_dict()


def evaluate(name, params, train, test_all, cv_df):
    t0 = time.perf_counter()
    model, n_rows = fit_model(name, train, params)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = predict_proba(model, test_all)
    pred_s = time.perf_counter() - t0
    v = test_all["label_valid"].to_numpy() == 1
    m = candidate_metrics(test_all.loc[v, "label"].to_numpy(), p[v])
    cv = cv_score(name, params, cv_df, config.CV_FOLDS)
    return model, p, {
        "cv_f1_macro": cv["f1_macro"], "cv_top1": cv["top1"], "cv_log_loss": cv["log_loss"],
        "test_top1": top1_accuracy(test_all, p), "test_f1_macro": m["f1_macro"],
        "test_precision_macro": m["precision_macro"], "test_recall_macro": m["recall_macro"],
        "test_log_loss": m["log_loss"], "fit_s": fit_s, "inference_s": pred_s, "train_rows": n_rows,
    }


def main(name, n_trials):
    for d in (OUT, FIG, config.MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    space = SEARCH_SPACES[name]
    (OUT / "search_space.json").write_text(json.dumps({"model": name, "space": space}, indent=2, default=str))

    train = valid(load_features("train"))
    test_all = load_features("student_test")
    tune_df = sample_observations(train, config.TUNE_OBS_FRAC, seed=config.RANDOM_SEED + 7).reset_index(drop=True)
    cv_df = sample_observations(train, config.CV_OBS_FRAC).reset_index(drop=True)  # same sample as script 07
    print(f"tuning {name}: {n_trials} trials, {config.TUNE_FOLDS}-fold GroupKFold on {len(tune_df)} rows")

    records = []

    def objective(trial):
        params = suggest(trial, space)
        t0 = time.perf_counter()
        s = cv_score(name, params, tune_df, config.TUNE_FOLDS)
        dur = time.perf_counter() - t0
        trial.set_user_attr("cv_f1_macro", s["f1_macro"])
        trial.set_user_attr("cv_log_loss", s["log_loss"])
        records.append({"trial": trial.number, **params, "cv_top1": s["top1"], "cv_f1_macro": s["f1_macro"],
                        "cv_log_loss": s["log_loss"], "seconds": dur})
        print(f"  trial {trial.number:3d}: top1={s['top1']:.4f} f1={s['f1_macro']:.4f} "
              f"({dur:.0f}s) {params}")
        return s["top1"]

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"pathsnap-{name}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # trial 0 = the untuned defaults, so the search starts from the baseline
    study.enqueue_trial({k: SPECS[name].defaults[k] for k in space if k in SPECS[name].defaults})
    t_search = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    search_s = time.perf_counter() - t_search

    trials = pd.DataFrame(records)
    trials["best_so_far"] = trials["cv_top1"].cummax()
    trials.to_csv(OUT / "trials.csv", index=False, float_format="%.5f")
    best_params = study.best_params
    (OUT / "best_params.json").write_text(json.dumps({
        "model": name, "best_params": best_params, "best_cv_top1": study.best_value,
        "objective": f"mean top-1 accuracy, {config.TUNE_FOLDS}-fold GroupKFold, "
                     f"{config.TUNE_OBS_FRAC:.0%} observation sample of valid train rows",
        "sampler": "Optuna TPESampler(seed=%d)" % config.RANDOM_SEED,
        "n_trials": n_trials, "search_seconds": search_s,
    }, indent=2, default=str))
    print(f"best: {best_params}  cv_top1={study.best_value:.4f}  ({search_s / 60:.1f} min)")

    print("refitting untuned and tuned models on all valid train rows ...")
    _, p_base, base = evaluate(name, {}, train, test_all, cv_df)
    tuned_model, p_tuned, tuned = evaluate(name, best_params, train, test_all, cv_df)
    cmp = pd.DataFrame([{"model": f"{name} (untuned)", **base}, {"model": f"{name} (tuned)", **tuned}])
    cmp.to_csv(OUT / "tuned_vs_untuned.csv", index=False, float_format="%.5f")
    print(cmp.round(4).T.to_string())

    joblib.dump(tuned_model, config.FINAL_MODEL_PATH, compress=3)
    size_mb = config.FINAL_MODEL_PATH.stat().st_size / 1e6
    (config.MODELS_DIR / "model_card.json").write_text(json.dumps({
        "model": name, "family": SPECS[name].family, "params": {**SPECS[name].defaults, **best_params},
        "features": FEATURES, "trained_on": "all valid train observations (label_valid == 1)",
        "train_rows": tuned["train_rows"], "random_seed": config.RANDOM_SEED,
        "sklearn_version": sklearn.__version__, "file_size_mb": round(size_mb, 1),
        "student_test": {k: tuned[k] for k in ("test_top1", "test_f1_macro", "test_log_loss")},
    }, indent=2, default=str))
    print(f"saved {config.FINAL_MODEL_PATH} ({size_mb:.1f} MB)")
    if size_mb > 95:
        print("WARNING: model file > 95 MB - GitHub rejects files over 100 MB. "
              "Re-run with a smaller n_estimators / larger min_samples_leaf range.")
    pd.DataFrame({"trip_id": test_all["trip_id"], "point_idx": test_all["point_idx"],
                  "arc_id": test_all["arc_id"], "proba": p_tuned}).to_csv(
        config.PROCESSED_DIR / "proba_student_test_final.csv.gz", index=False, float_format="%.6f")
    plots(trials, space, cmp)


def plots(trials, space, cmp):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(trials["trial"], trials["cv_top1"], color="#3b6ea5", s=22, label="trial")
    ax.plot(trials["trial"], trials["best_so_far"], color="#d0452c", label="best so far")
    ax.scatter([0], [trials.loc[0, "cv_top1"]], color="black", marker="x", s=60, label="trial 0 = defaults")
    ax.set_xlabel("trial")
    ax.set_ylabel("CV top-1 accuracy")
    ax.set_title("Bayesian optimisation history (TPE)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "optimisation_history.png", dpi=130); plt.close(fig)

    keys = list(space)
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.6), squeeze=False)
    for ax, k in zip(axes[0], keys):
        vals = trials[k]
        if space[k][0] == "cat":
            cats = [str(c) for c in space[k][1]]
            xs = vals.astype(str).map({c: i for i, c in enumerate(cats)})
            ax.set_xticks(range(len(cats)), cats)
        else:
            xs = vals
            if space[k][3]:
                ax.set_xscale("log")
        ax.scatter(xs, trials["cv_top1"], c=trials["trial"], cmap="viridis", s=22)
        ax.set_xlabel(k)
        ax.set_ylabel("CV top-1")
    fig.suptitle("Objective vs each hyperparameter (colour = trial number)")
    fig.tight_layout(); fig.savefig(FIG / "params_vs_objective.png", dpi=130); plt.close(fig)

    metrics = ["cv_f1_macro", "test_top1", "test_f1_macro", "test_log_loss"]
    fig, axes = plt.subplots(1, len(metrics) + 1, figsize=(15, 3.6))
    for ax, mname in zip(axes, metrics + ["inference_s"]):
        ax.bar([0, 1], cmp[mname], color=["#999999", "#3b6ea5"])
        ax.set_xticks([0, 1], ["untuned", "tuned"])
        for i, v in enumerate(cmp[mname]):
            ax.text(i, v, f"{v:.4f}" if mname != "inference_s" else f"{v:.1f}s", ha="center", va="bottom",
                    fontsize=8)
        ax.set_title(mname, fontsize=10)
        lo = cmp[mname].min()
        if mname not in ("test_log_loss", "inference_s"):
            ax.set_ylim(max(0, lo - 0.05), min(1.0, cmp[mname].max() + 0.02))
    fig.suptitle("Untuned vs tuned (same data, same metrics)")
    fig.tight_layout(); fig.savefig(FIG / "tuned_vs_untuned.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--trials", type=int, default=config.TUNE_TRIALS)
    a = ap.parse_args()
    name = a.model or json.loads((config.RESULTS_DIR / "models" / "selected_model.json").read_text())["selected"]
    main(name, a.trials)
