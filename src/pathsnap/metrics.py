"""Evaluation metrics for candidate classification and route reconstruction.

Two levels are reported, as the guidelines ask:

* candidate level (binary, one row per candidate): accuracy, macro
  precision / recall / F1 at threshold 0.5, log loss, confusion matrix;
* observation / route level: top-1 accuracy (highest-scored candidate per
  observation vs the reference arc), candidate miss rate, route
  continuity, and arc-set precision / recall / F1 per trip.

**Primary metric = observation-level top-1 accuracy** on valid
observations: it measures exactly what PathSnap has to output (one arc per
GPS fix) and is insensitive to the 0.5 threshold, unlike F1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["trip_id", "point_idx"]


def candidate_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
    pred = (p >= threshold).astype(int)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "log_loss": log_loss(y, pc, labels=[0, 1]),
    }


def confusion(y, p, threshold=0.5):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1])


def pick_top1(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """Highest score per observation; ties go to the nearer candidate (lower dist_rank)."""
    d = df[KEY + ["arc_id", "dist_rank", "label"]].copy()
    d["score"] = score
    d = d.sort_values(KEY + ["score", "dist_rank"], ascending=[True, True, False, True], kind="stable")
    return d.drop_duplicates(KEY, keep="first")


def top1_accuracy(df: pd.DataFrame, score: np.ndarray) -> float:
    """Over valid observations; an observation whose reference arc is not a candidate counts as wrong."""
    v = df["label_valid"].to_numpy() == 1
    sub = df[v]
    n_obs = sub[KEY].drop_duplicates().shape[0]
    best = pick_top1(sub, score[v])
    return float(best["label"].sum() / max(n_obs, 1))


def candidate_miss_rate(df: pd.DataFrame) -> float:
    sub = df[df["label_valid"] == 1]
    hit = sub.groupby(KEY)["label"].max()
    return float(1 - hit.mean())


def route_continuity(arcs: np.ndarray, u: np.ndarray, v: np.ndarray, max_hops: int = 1) -> tuple[float, int, int]:
    """Share of arc changes between consecutive predictions that follow the network.

    A change a -> b counts as continuous when b can be entered within
    ``max_hops`` arcs of leaving a: max_hops = 1 means b starts exactly where
    a ends (v[a] == u[b]); max_hops = 3 also accepts passing through up to two
    very short arcs within one GPS step. Returns (continuity, n_changes, n_jumps).
    """
    arcs = np.asarray(arcs)
    if len(arcs) < 2:
        return np.nan, 0, 0
    a, b = arcs[:-1], arcs[1:]
    change = a != b
    n = int(change.sum())
    if n == 0:
        return 1.0, 0, 0
    ok = v[a[change]] == u[b[change]]
    if max_hops > 1 and not ok.all():
        order = np.argsort(u, kind="stable")
        starts = np.searchsorted(u[order], np.arange(max(u.max(), v.max()) + 2))
        for k in np.flatnonzero(~ok):
            src, dst = v[a[change][k]], u[b[change][k]]
            frontier, seen = {int(src)}, {int(src)}
            for _ in range(max_hops - 1):
                nxt = set()
                for node in frontier:
                    for arc in order[starts[node]:starts[node + 1]]:
                        w = int(v[arc])
                        if w not in seen:
                            seen.add(w)
                            nxt.add(w)
                if dst in nxt:
                    ok[k] = True
                    break
                frontier = nxt
    return float(ok.mean()), n, int((~ok).sum())


def arc_set_scores(pred_arcs: np.ndarray, true_arcs: np.ndarray) -> dict:
    """Sequence-level view: does the set of arcs we output match the reference arcs?"""
    P, T = set(np.asarray(pred_arcs).tolist()), set(np.asarray(true_arcs).tolist())
    inter = len(P & T)
    prec = inter / len(P) if P else 0.0
    rec = inter / len(T) if T else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"arc_precision": prec, "arc_recall": rec, "arc_f1": f1}
