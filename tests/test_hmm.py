"""Unit tests for the HMM / Viterbi decoder on a hand-built network.

Run:  python tests/test_hmm.py      (or: python -m pytest tests)

Network (metres, local plane), all arcs 100 m:

    node0 --a0--> node1 --a1--> node2          main road, eastbound
    node1 --a2--> node3                        side street going north
    node3 --a4--> node1                        side street coming back south
    node9 --a3--> node8                        a disconnected far-away arc
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pathsnap.geo import LocalProjection
from pathsnap.hmm import HMMParams, RouteDistance, decode_arcs, precompute
from pathsnap.metrics import route_continuity
from pathsnap.network import RoadNetwork


class FlatProjection(LocalProjection):
    def __init__(self):
        super().__init__(0, 0)

    def to_xy(self, lon, lat):
        return np.asarray(lon, float), np.asarray(lat, float)


def network():
    nodes = pd.DataFrame({"lon": [0, 100, 200, 100, 0, 0, 0, 0, 5000, 5100],
                          "lat": [0, 0, 0, 100, 0, 0, 0, 0, 5000, 5000]})
    arcs = pd.DataFrame({"u": [0, 1, 1, 9, 3], "v": [1, 2, 3, 8, 1]})
    return RoadNetwork(nodes, arcs, FlatProjection())


def test_route_distance():
    rd = RouteDistance(network())
    # from the middle of a0 to 30 m into a1 = 50 + 30
    d = rd.from_position(0, 0.5, {1: [(0, 0.3)]}, limit=500)
    assert abs(d[0] - 80) < 1e-9
    # same arc, moving forward 20 m
    d = rd.from_position(1, 0.2, {1: [(0, 0.4)]}, limit=500)
    assert abs(d[0] - 20) < 1e-9
    # the disconnected arc is unreachable
    assert rd.from_position(0, 0.5, {3: [(0, 0.5)]}, limit=500) == {}


def test_viterbi_fixes_a_noisy_classifier_choice():
    """The classifier prefers the side street (a2) at t=2, but the vehicle
    ends up on a1, and the side street plus a1 is not drivable within one
    second, so Viterbi must keep the whole sequence on the main road."""
    net = network()
    xy = np.array([[40, 2], [60, 2], [99, 4], [120, 2], [140, 2]], float)
    arcs = [np.array([0]), np.array([0]), np.array([2, 0]), np.array([1]), np.array([1])]
    t_al = [np.array([0.4]), np.array([0.6]), np.array([0.04, 0.99]), np.array([0.2]), np.array([0.4])]
    prob = [np.array([0.9]), np.array([0.9]), np.array([0.7, 0.3]), np.array([0.9]), np.array([0.9])]
    inp = precompute(arcs, t_al, prob, xy, np.zeros(5, int), net, max_detour_m=150)
    hmm, clf, breaks = decode_arcs(inp, HMMParams(beta_m=5, emission_weight=1))
    assert clf.tolist() == [0, 0, 2, 1, 1]          # independent choices jump onto the side street
    assert hmm.tolist() == [0, 0, 0, 1, 1]          # the decoder stays on the road actually driven
    assert breaks == 0
    u, v = net.u, net.v
    assert route_continuity(hmm, u, v)[0] == 1.0
    assert route_continuity(clf, u, v)[0] < 1.0


def test_gap_and_missing_candidates_break_the_chain():
    net = network()
    xy = np.array([[40, 2], [60, 2], [5050, 5000], [5060, 5000]], float)
    arcs = [np.array([0]), np.array([], int), np.array([3]), np.array([3])]
    t_al = [np.array([0.4]), np.array([]), np.array([0.5]), np.array([0.6])]
    prob = [np.array([0.9]), np.array([]), np.array([0.8]), np.array([0.8])]
    inp = precompute(arcs, t_al, prob, xy, np.array([0, 0, 1, 1]), net)
    hmm, _, _ = decode_arcs(inp, HMMParams())
    assert hmm.tolist() == [0, -1, 3, 3]


def test_unreachable_step_counts_as_a_break():
    net = network()
    xy = np.array([[40, 2], [5050, 5000]], float)
    arcs = [np.array([0]), np.array([3])]
    inp = precompute(arcs, [np.array([0.4]), np.array([0.5])], [np.array([0.9]), np.array([0.9])], xy,
                     np.array([0, 0]), net)
    hmm, _, breaks = decode_arcs(inp, HMMParams())
    assert hmm.tolist() == [0, 3] and breaks == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("all HMM tests passed")
