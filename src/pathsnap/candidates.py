"""Candidate generation: which road arcs could each GPS observation be on?

Map matching is framed as *candidate classification*: for each observation
we collect nearby arcs (the candidates) and a classifier scores how likely
each one is the true arc. The same candidate sets later become the hidden
states of the HMM, so the candidate -> arc_id link is kept explicitly.

Rule:
* every arc within CANDIDATE_RADIUS_M of the observation, nearest first,
  capped at CANDIDATE_MAX_K;
* if an observation has no arc in that radius (GPS outage, off-network),
  fall back to the nearest arcs within CANDIDATE_FALLBACK_RADIUS_M so the
  decoder always has at least one state where possible.

The radius / K values were chosen from the train-partition recall curve
(``scripts/05_build_features.py`` -> results/features/candidate_recall.csv).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .network import RoadNetwork


def _topk(point_idx, arc_id, dist, t, k):
    df = pd.DataFrame({"point_idx": point_idx, "arc_id": arc_id, "dist_m": dist, "t_along": t})
    df = df.sort_values(["point_idx", "dist_m", "arc_id"], kind="stable")
    df["rank"] = df.groupby("point_idx").cumcount()
    return df[df["rank"] < k]


def generate_candidates(x: np.ndarray, y: np.ndarray, net: RoadNetwork,
                        radius: float = config.CANDIDATE_RADIUS_M,
                        k: int = config.CANDIDATE_MAX_K,
                        fallback_radius: float = config.CANDIDATE_FALLBACK_RADIUS_M) -> pd.DataFrame:
    pi, ai, d, t = net.query_radius(x, y, radius)
    cands = _topk(pi, ai, d, t, k)
    cands["fallback"] = 0

    missing = np.setdiff1d(np.arange(len(x)), cands["point_idx"].unique())
    if len(missing) and fallback_radius > radius:
        pi2, ai2, d2, t2 = net.query_radius(x[missing], y[missing], fallback_radius)
        fb = _topk(missing[pi2], ai2, d2, t2, k)
        fb["fallback"] = 1
        cands = pd.concat([cands, fb], ignore_index=True)
        cands = cands.sort_values(["point_idx", "rank"], kind="stable")
    return cands.reset_index(drop=True)


def candidate_recall(cands: pd.DataFrame, true_arc: np.ndarray, valid: np.ndarray) -> dict:
    """How often the reference arc is inside the candidate set (valid labels only)."""
    hit = np.zeros(len(true_arc), dtype=bool)
    m = cands["arc_id"].to_numpy() == true_arc[cands["point_idx"].to_numpy()]
    hit[cands["point_idx"].to_numpy()[m]] = True
    n_valid = int(valid.sum())
    return {
        "n_valid_obs": n_valid,
        "candidate_recall": float(hit[valid].mean()) if n_valid else np.nan,
        "candidate_miss_rate": float(1 - hit[valid].mean()) if n_valid else np.nan,
        "obs_without_candidates": int(len(true_arc) - cands["point_idx"].nunique()),
        "mean_candidates_per_obs": float(len(cands) / max(len(true_arc), 1)),
    }
