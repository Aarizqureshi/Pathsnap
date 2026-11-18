"""HMM / Viterbi route reconstruction: tune the decoder, then compare
classifier-only vs classifier + HMM.

Decoder tuning (train data only)
  * HMM_TUNE_HOLDOUT_TRIPS train trips are held out; the final model's
    configuration is refitted on the other train trips, so the held-out
    probabilities are genuinely out-of-sample.
  * Grid over beta_m x emission_weight; choose the highest pooled top-1 on
    the held-out trips (tie: fewer 3-hop discontinuities).
  * Saved to configs/hmm.json (read by the API).

Student-test comparison
  * models/final_model.joblib (from script 08) + the chosen decoder.
  * Per trip: top-1, strict and 3-hop continuity, discontinuities, arc-set F1,
    for classifier-only and HMM; plus the trips where HMM did not help.

Outputs
  configs/hmm.json, results/hmm/*.csv, results/figures/hmm/*.png,
  results/predictions/<trip>.csv (used by scripts/10_prediction_maps.py)

Usage:  python scripts/09_hmm_decode.py
"""
import itertools
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

from pathsnap import config
from pathsnap.evaluation import evaluate_trip
from pathsnap.hmm import HMMParams
from pathsnap.inference import Predictor
from pathsnap.io import list_trip_dirs, load_trip
from pathsnap.models import fit_model, load_features, valid

OUT = config.RESULTS_DIR / "hmm"
FIG = config.FIGURES_DIR / "hmm"
PRED = config.RESULTS_DIR / "predictions"


def pooled(rows, name):
    df = pd.DataFrame(rows)
    w = df["n_valid_labels"]
    return {
        f"top1_{name}": float((df[f"top1_{name}"] * w).sum() / w.sum()),
        f"continuity_{name}": 1 - df[f"discontinuities_{name}"].sum() / max(df[f"arc_changes_{name}"].sum(), 1),
        f"continuity_3hop_{name}": 1 - df[f"discontinuities_3hop_{name}"].sum()
        / max(df[f"arc_changes_{name}"].sum(), 1),
        f"discontinuities_{name}": int(df[f"discontinuities_{name}"].sum()),
        f"arc_f1_{name}": float(df[f"arc_f1_{name}"].mean()),
    }


def tune_decoder(card):
    trip_ids = [d.name for d in list_trip_dirs("train")]
    rng = np.random.default_rng(config.RANDOM_SEED)
    holdout = sorted(rng.choice(trip_ids, size=config.HMM_TUNE_HOLDOUT_TRIPS, replace=False).tolist())
    train = valid(load_features("train"))
    fit_part = train[~train["trip_id"].isin(holdout)]
    print(f"decoder tuning: refitting {card['model']} on {fit_part['trip_id'].nunique()} train trips, "
          f"holding out {len(holdout)}")
    model, _ = fit_model(card["model"], fit_part, card["params"])
    pr = Predictor(model=model, hmm_params=HMMParams())

    cache = []
    for tid in holdout:
        trip = load_trip(config.DATA_DIR / "train" / tid, "train")
        cache.append((trip, pr.score_trip(trip)))

    grid = []
    for beta, w in itertools.product(config.HMM_GRID["beta_m"], config.HMM_GRID["emission_weight"]):
        params = HMMParams(beta_m=beta, emission_weight=w)
        rows = [evaluate_trip(trip, pr.decode(sc, params)) for trip, sc in cache]
        g = {"beta_m": beta, "emission_weight": w, **pooled(rows, "hmm"), **pooled(rows, "classifier")}
        grid.append(g)
        print(f"  beta={beta:5.1f} w={w:3.1f}: top1 hmm={g['top1_hmm']:.4f} "
              f"(classifier {g['top1_classifier']:.4f}) 3-hop continuity={g['continuity_3hop_hmm']:.4f}")
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "decoder_grid.csv", index=False, float_format="%.5f")
    best = grid.sort_values(["top1_hmm", "continuity_3hop_hmm"], ascending=False).iloc[0]
    params = HMMParams(beta_m=float(best["beta_m"]), emission_weight=float(best["emission_weight"]))
    config.CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    config.HMM_CONFIG_PATH.write_text(json.dumps({
        "params": params.to_dict(),
        "tuned_on": f"{len(holdout)} held-out train trips, model refitted on the remaining train trips",
        "holdout_trips": holdout,
        "selection": "highest pooled top-1 (HMM) on held-out trips, tie-break 3-hop continuity",
        "holdout_top1_hmm": float(best["top1_hmm"]), "holdout_top1_classifier": float(best["top1_classifier"]),
    }, indent=2))
    print(f"chosen decoder: {params}")

    piv = grid.pivot(index="beta_m", columns="emission_weight", values="top1_hmm")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(piv.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.set_xlabel("emission weight w")
    ax.set_ylabel("beta (m)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.iat[i, j]:.4f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"Decoder grid: HMM top-1 on {len(holdout)} held-out train trips")
    fig.tight_layout(); fig.savefig(FIG / "decoder_grid.png", dpi=130); plt.close(fig)
    return params


def student_test(params):
    pr = Predictor(model=joblib.load(config.FINAL_MODEL_PATH), hmm_params=params)
    rows = []
    for d in list_trip_dirs("student_test"):
        trip = load_trip(d, "student_test")
        t0 = time.perf_counter()
        pred, info = pr.predict_trip(trip)
        total = time.perf_counter() - t0
        ev = evaluate_trip(trip, pred)
        rows.append({"trip_id": trip.trip_id, **ev, "hmm_breaks": info["hmm_breaks"], "predict_s": total,
                     **{k: v for k, v in info["timings"].items()}})
        pred.to_csv(PRED / f"{trip.trip_id}.csv", index=False, float_format="%.7f")
        r = rows[-1]
        print(f"  {trip.trip_id}: top1 {r['top1_classifier']:.4f} -> {r['top1_hmm']:.4f}  "
              f"jumps {r['discontinuities_classifier']} -> {r['discontinuities_hmm']}  ({total:.1f}s)")
    per = pd.DataFrame(rows)
    per["top1_gain"] = per["top1_hmm"] - per["top1_classifier"]
    per.to_csv(OUT / "student_test_per_trip.csv", index=False, float_format="%.5f")
    summ = pd.DataFrame([
        {"method": "classifier only", **{k.replace("_classifier", ""): v for k, v in pooled(rows, "classifier").items()}},
        {"method": "classifier + HMM", **{k.replace("_hmm", ""): v for k, v in pooled(rows, "hmm").items()}},
    ])
    summ.to_csv(OUT / "student_test_summary.csv", index=False, float_format="%.5f")
    print(summ.round(4).to_string(index=False))
    worse = per[per["top1_gain"] < 0][["trip_id", "top1_classifier", "top1_hmm", "top1_gain"]]
    worse.to_csv(OUT / "trips_hmm_did_not_help.csv", index=False, float_format="%.5f")
    print(f"trips where HMM lowered top-1: {len(worse)}")

    x = np.arange(len(per))
    w = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(x - w / 2, per["top1_classifier"], w, label="classifier only", color="#999999")
    axes[0].bar(x + w / 2, per["top1_hmm"], w, label="classifier + HMM", color="#3b6ea5")
    axes[0].set_ylabel("top-1 accuracy")
    axes[0].set_ylim(max(0, per[["top1_classifier", "top1_hmm"]].min().min() - 0.05), 1.0)
    axes[0].legend()
    axes[1].bar(x - w / 2, per["discontinuities_classifier"], w, label="classifier only", color="#999999")
    axes[1].bar(x + w / 2, per["discontinuities_hmm"], w, label="classifier + HMM", color="#3b6ea5")
    axes[1].set_ylabel("route discontinuities (strict)")
    axes[1].set_xticks(x, per["trip_id"].str[-3:])
    axes[1].set_xlabel("student-test trip")
    fig.suptitle("Student-test: effect of HMM/Viterbi decoding per trip")
    fig.tight_layout(); fig.savefig(FIG / "classifier_vs_hmm_per_trip.png", dpi=130); plt.close(fig)


def main():
    for d in (OUT, FIG, PRED):
        d.mkdir(parents=True, exist_ok=True)
    card = json.loads((config.MODELS_DIR / "model_card.json").read_text())
    params = tune_decoder(card)
    student_test(params)


if __name__ == "__main__":
    main()
