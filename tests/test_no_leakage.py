"""Leakage test: a trip's predictions must not depend on its reference route.

For every trip the full prediction pipeline (clean -> candidates -> features
-> classifier -> HMM) is run four times:

  a) with the real reference route attached to the trip
  b) with no route at all
  c) with the route replaced by random arc ids
  d) with a "poisoned" route that raises an error if anything touches it

All four must give identical arcs for every GPS observation, for both the
classifier-only and the HMM output. If (d) raises, some code read the
route during prediction. The test never scores anything, so it reveals no
accuracy.

Run:  python tests/test_no_leakage.py [--partition student_test] [--max-trips 10]
"""
import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from pathsnap import config
from pathsnap.inference import Predictor
from pathsnap.io import list_trip_dirs, load_trip


class PoisonedRoute:
    """Stands in for the route; any use of it is a leak."""

    def _leak(self, *a, **k):
        raise AssertionError("LEAK: the reference route was accessed during prediction")

    __getitem__ = __len__ = __iter__ = __array__ = __getattr__ = __bool__ = _leak


def check_trip(trip, predictor) -> tuple[bool, str]:
    rng = np.random.default_rng(0)
    n_arcs = len(trip.arcs)
    route_len = len(trip.route) if trip.route is not None else 50
    variants = {
        "real route": trip,
        "no route": dataclasses.replace(trip, route=None),
        "random route": dataclasses.replace(trip, route=rng.integers(0, n_arcs, route_len)),
        "poisoned route": dataclasses.replace(trip, route=PoisonedRoute()),
    }
    outputs = {}
    for name, t in variants.items():
        try:
            pred, _ = predictor.predict_trip(t)
        except AssertionError as e:
            return False, f"{name}: {e}"
        outputs[name] = (pred["arc_classifier"].to_numpy(), pred["arc_hmm"].to_numpy())
    ref = outputs["real route"]
    for name, (clf, hmm) in outputs.items():
        if not (np.array_equal(clf, ref[0]) and np.array_equal(hmm, ref[1])):
            return False, f"predictions with '{name}' differ from predictions with the real route"
    return True, f"{len(ref[1])} observations, identical in all 4 variants"


def test_student_test_trips():
    """pytest entry point: first 3 student-test trips with the final model."""
    if not config.FINAL_MODEL_PATH.exists():
        return
    pr = Predictor()
    for d in list_trip_dirs("student_test")[:3]:
        ok, msg = check_trip(load_trip(d, "student_test"), pr)
        assert ok, msg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--max-trips", type=int, default=10)
    a = ap.parse_args()
    pr = Predictor()
    failed = 0
    for d in list_trip_dirs(a.partition)[:a.max_trips]:
        ok, msg = check_trip(load_trip(d, a.partition), pr)
        failed += not ok
        print(("ok    " if ok else "FAIL  ") + d.name + "  " + msg)
    print("NO LEAKAGE: predictions are independent of the reference route" if not failed
          else f"{failed} trip(s) FAILED")
    sys.exit(1 if failed else 0)
