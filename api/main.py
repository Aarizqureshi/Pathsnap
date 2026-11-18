"""PathSnap FastAPI backend (laptop 1).

Endpoints
  GET  /health                       model + decoder status
  GET  /partitions                   test partitions the UI may offer (never train)
  GET  /trips?partition=...          trip ids in one partition
  GET  /model-info                   final model, decoder and its development accuracy
  GET  /                             redirects to /ui/: the evaluator UI served by this same server, so
                                     laptop 2 only needs a browser: http://<laptop-1 IP>:8000/
  POST /predict                      {"trip_id": "...", "partition": "..."}
                                     -> predicted route + server-side metrics
                                        + running overall accuracy for that partition

Flow for /predict
  1. load the trip's raw files from disk (track, nodes, arcs, route)
  2. prediction pipeline on a copy WITHOUT the route:
     clean -> candidates -> features -> classifier -> HMM/Viterbi
  3. only then score the prediction against the reference route (server side)
  4. return GPS points, predicted route geometry and aggregate metrics.
     Reference arcs are never sent to the client.

Run (from the repository root):
  set PATHSNAP_DATA=<folder containing student_test\\ and evaluator_test\\>
  set PATHSNAP_PARTITIONS=student_test,evaluator_test     (default)
  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pathsnap import config
from pathsnap.evaluation import evaluate_trip
from pathsnap.inference import Predictor
from pathsnap.io import list_trip_dirs, load_trip

REQUESTED = [p.strip() for p in os.environ.get("PATHSNAP_PARTITIONS", "student_test,evaluator_test").split(",")
             if p.strip()]
FORBIDDEN = {"train"}  # training trips are never served
UI_DIR = Path(__file__).resolve().parents[1] / "ui"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap.api")

_predictor: Predictor | None = None
# per partition: {trip_id: {"n_valid", "n_correct_classifier", "n_correct_hmm"}} for trips run since startup
_session: dict[str, dict[str, dict]] = {}


def predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        t0 = time.perf_counter()
        _predictor = Predictor()
        log.info("loaded model %s and decoder %s in %.1fs", config.FINAL_MODEL_PATH.name,
                 _predictor.params.to_dict(), time.perf_counter() - t0)
    return _predictor


def partitions() -> list[str]:
    return [p for p in REQUESTED if p not in FORBIDDEN and (config.DATA_DIR / p).is_dir()]


def trips_in(partition: str) -> list[str]:
    if partition not in partitions():
        raise HTTPException(status_code=404, detail=f"unknown partition {partition}")
    return [d.name for d in list_trip_dirs(partition)]


def clean_number(v):
    """JSON has no NaN: send null instead, round floats for readability."""
    if isinstance(v, float):
        return None if math.isnan(v) else round(v, 5)
    return v


def development_accuracy() -> dict | None:
    """Student-test accuracy of the deployed model, as measured by scripts/09_hmm_decode.py."""
    path = config.RESULTS_DIR / "hmm" / "student_test_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path).set_index("method")
    pick = lambda m: {k: clean_number(float(df.loc[m, k])) for k in ("top1", "continuity", "continuity_3hop")}
    return {"partition": "student_test", "classifier_only": pick("classifier only"),
            "classifier_hmm": pick("classifier + HMM")}


def session_summary(partition: str) -> dict:
    runs = _session.get(partition, {})
    n = sum(r["n_valid"] for r in runs.values())
    c_clf = sum(r["n_correct_classifier"] for r in runs.values())
    c_hmm = sum(r["n_correct_hmm"] for r in runs.values())
    return {"partition": partition, "trips_run": len(runs), "n_valid": n,
            "n_correct_classifier": c_clf, "n_correct_hmm": c_hmm,
            "top1_classifier": clean_number(c_clf / n) if n else None,
            "top1_hmm": clean_number(c_hmm / n) if n else None}


def polylines(trip, arc_seq):
    """Connected runs of the predicted arc sequence as [[lat, lon], ...] lists."""
    lon, lat = trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy()
    u, v = trip.arcs["u"].to_numpy(), trip.arcs["v"].to_numpy()
    seq = [int(a) for a in arc_seq if a >= 0]
    seq = [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]
    runs, cur = [], []
    for i, a in enumerate(seq):
        if cur and u[a] != v[seq[i - 1]]:
            runs.append(cur)
            cur = []
        if not cur:
            cur.append([round(float(lat[u[a]]), 7), round(float(lon[u[a]]), 7)])
        cur.append([round(float(lat[v[a]]), 7), round(float(lon[v[a]]), 7)])
    if cur:
        runs.append(cur)
    return runs


def _startup():
    log.info("PathSnap API starting: data=%s", config.DATA_DIR)
    for p in REQUESTED:
        if p in FORBIDDEN:
            log.warning("partition %s is never served", p)
        elif not (config.DATA_DIR / p).is_dir():
            log.warning("partition %s not found under %s", p, config.DATA_DIR)
        else:
            log.info("partition %s: %d trips", p, len(list_trip_dirs(p)))
    predictor()


@asynccontextmanager
async def lifespan(_app):
    _startup()
    yield


app = FastAPI(title="PathSnap API", version="1.1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")


class PredictRequest(BaseModel):
    trip_id: str
    partition: str = "student_test"


@app.get("/health")
def health():
    p = predictor()
    return {"status": "ok", "partitions": partitions(), "model": config.FINAL_MODEL_PATH.name,
            "decoder": p.params.to_dict()}


@app.get("/partitions")
def get_partitions():
    return {"partitions": partitions()}


@app.get("/trips")
def trips(partition: str = "student_test"):
    ids = trips_in(partition)
    log.info("GET /trips partition=%s -> %d trips", partition, len(ids))
    return {"partition": partition, "trips": ids}


@app.get("/model-info")
def model_info():
    card_path = config.MODELS_DIR / "model_card.json"
    card = json.loads(card_path.read_text()) if card_path.exists() else {}
    return {"model": card.get("model"), "family": card.get("family"), "params": card.get("params"),
            "train_rows": card.get("train_rows"), "decoder": predictor().params.to_dict(),
            "development_accuracy": development_accuracy()}


@app.post("/predict")
def predict(req: PredictRequest):
    t_start = time.perf_counter()
    if req.trip_id not in trips_in(req.partition):  # also blocks path traversal
        log.warning("POST /predict unknown trip_id=%r partition=%s", req.trip_id, req.partition)
        raise HTTPException(status_code=404, detail=f"unknown trip_id {req.trip_id}")
    log.info("POST /predict partition=%s trip_id=%s received", req.partition, req.trip_id)

    trip = load_trip(config.DATA_DIR / req.partition / req.trip_id, req.partition)
    pred, info = predictor().predict_trip(trip)          # route is stripped inside
    t_pred = time.perf_counter() - t_start
    metrics = evaluate_trip(trip, pred)                   # server-side scoring only
    total = time.perf_counter() - t_start

    log.info("trip %s: %d observations, %d HMM breaks, prediction %.2fs, total %.2fs",
             req.trip_id, len(pred), info["hmm_breaks"], t_pred, total)
    if metrics.get("available"):
        log.info("trip %s: top-1 classifier=%.4f (%d/%d) HMM=%.4f (%d/%d) | continuity classifier=%.4f HMM=%.4f",
                 req.trip_id, metrics["top1_classifier"], metrics["n_correct_classifier"],
                 metrics["n_valid_labels"], metrics["top1_hmm"], metrics["n_correct_hmm"],
                 metrics["n_valid_labels"], metrics["continuity_classifier"], metrics["continuity_hmm"])
        _session.setdefault(req.partition, {})[req.trip_id] = {
            "n_valid": metrics["n_valid_labels"], "n_correct_classifier": metrics["n_correct_classifier"],
            "n_correct_hmm": metrics["n_correct_hmm"]}
    overall = session_summary(req.partition)
    if overall["trips_run"]:
        log.info("overall %s so far: %d trips, top-1 classifier=%.4f HMM=%.4f", req.partition,
                 overall["trips_run"], overall["top1_classifier"], overall["top1_hmm"])

    return {
        "trip_id": req.trip_id,
        "partition": req.partition,
        "n_observations": int(len(pred)),
        "gps": pred[["lat", "lon"]].round(7).to_numpy().tolist(),
        "predicted_route": polylines(trip, pred["arc_hmm"].to_numpy()),
        "predicted_arcs": pred["arc_hmm"].astype(int).tolist(),
        "metrics": {k: clean_number(v) for k, v in metrics.items()},
        "overall": overall,
        "hmm_breaks": int(info["hmm_breaks"]),
        "timings_s": {k: round(v, 3) for k, v in {**info["timings"], "prediction_total": t_pred,
                                                  "request_total": total}.items()},
    }


# Serve the evaluator UI from this server (same origin: no CORS, no second web server needed).
if UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
