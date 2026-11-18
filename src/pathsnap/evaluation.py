"""Server-side scoring of a prediction against the reference route.

Kept apart from :mod:`pathsnap.inference` on purpose: the reference route
is only read here, after the prediction is finished, and only aggregate
metrics leave this module (never the reference arcs themselves).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import Trip
from .metrics import arc_set_scores, route_continuity
from .pipeline import prepare_trip


def evaluate_trip(trip: Trip, pred: pd.DataFrame) -> dict:
    if trip.route is None:
        return {"available": False}
    ref = prepare_trip(trip, with_labels=True).points
    if len(ref) != len(pred) or not np.array_equal(ref["obs_idx"].to_numpy(), pred["obs_idx"].to_numpy()):
        raise RuntimeError("prediction and reference observations do not line up")
    true = ref["true_arc"].to_numpy()
    valid = ref["label_valid"].to_numpy() == 1
    u, v = trip.arcs["u"].to_numpy(), trip.arcs["v"].to_numpy()
    out = {"available": True, "n_observations": int(len(ref)), "n_valid_labels": int(valid.sum())}
    for name, col in (("classifier", "arc_classifier"), ("hmm", "arc_hmm")):
        p = pred[col].to_numpy()
        have = p >= 0
        cont, changes, jumps = route_continuity(p[have], u, v, max_hops=1)
        cont3, _, jumps3 = route_continuity(p[have], u, v, max_hops=3)
        n_correct = int((p[valid] == true[valid]).sum())
        out[f"n_correct_{name}"] = n_correct
        out[f"top1_{name}"] = n_correct / int(valid.sum()) if valid.any() else np.nan
        out[f"continuity_{name}"] = cont
        out[f"continuity_3hop_{name}"] = cont3
        out[f"discontinuities_{name}"] = jumps
        out[f"discontinuities_3hop_{name}"] = jumps3
        out[f"arc_changes_{name}"] = changes
        out.update({f"{k}_{name}": s for k, s in arc_set_scores(p[valid & have], true[valid]).items()})
    return out
