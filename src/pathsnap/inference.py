"""End-to-end prediction for one trip: raw files -> arc per GPS observation.

Used by the HMM experiments and by the API, so both run the exact same
code path. The reference route is removed from the trip *before* anything
else happens: prediction can never see labels.
"""
from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .candidates import generate_candidates
from .features import build_features
from .hmm import DecodeInput, HMMParams, decode_arcs, precompute
from .io import Trip
from .network import RoadNetwork
from .pipeline import prepare_trip


def strip_labels(trip: Trip) -> Trip:
    return dataclasses.replace(trip, route=None)


def load_hmm_params(path=config.HMM_CONFIG_PATH) -> HMMParams:
    if path.exists():
        return HMMParams.from_dict(json.loads(path.read_text())["params"])
    return HMMParams()


@dataclass
class TripScores:
    trip_id: str
    points: pd.DataFrame
    net: RoadNetwork
    decode_input: DecodeInput
    timings: dict


class Predictor:
    def __init__(self, model=None, hmm_params: HMMParams | None = None):
        if model is None:
            import joblib
            model = joblib.load(config.FINAL_MODEL_PATH)
        self.model = model
        self.params = hmm_params or load_hmm_params()

    def score_trip(self, trip: Trip) -> TripScores:
        tm = {}
        t0 = time.perf_counter()
        pt = prepare_trip(strip_labels(trip), with_labels=False)
        x, y = pt.xy
        tm["preprocess_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        cands = generate_candidates(x, y, pt.net)
        feats = build_features(pt.points, cands, pt.net)
        tm["candidates_features_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        proba = self.model.predict_proba(feats[config.FEATURE_COLUMNS].to_numpy(np.float64))[:, 1]
        tm["classifier_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        n = len(pt.points)
        p_idx = cands["point_idx"].to_numpy()
        bounds = np.searchsorted(p_idx, np.arange(n + 1))  # cands are sorted by point_idx
        arc_arr, t_arr = cands["arc_id"].to_numpy(), cands["t_along"].to_numpy()
        arcs = [arc_arr[bounds[i]:bounds[i + 1]] for i in range(n)]
        t_al = [t_arr[bounds[i]:bounds[i + 1]] for i in range(n)]
        prob = [proba[bounds[i]:bounds[i + 1]] for i in range(n)]
        inp = precompute(arcs, t_al, prob, np.c_[x, y], pt.points["segment_id"].to_numpy(), pt.net,
                         self.params.max_detour_m)
        tm["route_distances_s"] = time.perf_counter() - t0
        return TripScores(trip.trip_id, pt.points, pt.net, inp, tm)

    def decode(self, scores: TripScores, params: HMMParams | None = None) -> pd.DataFrame:
        params = params or self.params
        t0 = time.perf_counter()
        hmm, clf, breaks = decode_arcs(scores.decode_input, params)
        scores.timings["viterbi_s"] = time.perf_counter() - t0
        out = scores.points[["obs_idx", "t", "lon", "lat", "segment_id"]].copy()
        out.insert(0, "point_idx", np.arange(len(out)))
        out["arc_classifier"] = clf
        out["arc_hmm"] = hmm
        out.attrs["hmm_breaks"] = breaks
        return out

    def predict_trip(self, trip: Trip) -> tuple[pd.DataFrame, dict]:
        scores = self.score_trip(trip)
        pred = self.decode(scores)
        info = {"timings": scores.timings, "hmm_breaks": pred.attrs["hmm_breaks"],
                "u": scores.net.u, "v": scores.net.v}
        return pred, info
