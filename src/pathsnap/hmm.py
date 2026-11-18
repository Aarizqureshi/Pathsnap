"""Hidden Markov Model + Viterbi decoding over candidate arcs.

Hidden states   at observation t: the candidate arcs of that observation
                (the same candidates the classifier scored, so the
                state -> arc_id correspondence is preserved).
Emission        log P(o_t | arc) = w * log(max(p_classifier, min_prob))
Transition      Newson & Krumm (2009): a vehicle's driven distance between two
                fixes is close to the straight-line distance between them, so
                log P(arc_i -> arc_j) = -| d_route(i, j) - d_straight | / beta
                where d_route is the shortest network distance from the
                projection of fix t-1 on arc i to the projection of fix t on
                arc j (bounded Dijkstra over the directed arc graph).
                Unreachable within d_straight + max_detour -> impossible.
Viterbi         delta_t(j) = max_i [delta_{t-1}(i) + log A(i, j)] + log B_t(j)
                with back-pointers, then back-tracking from the best final state.

Breaks: a time gap (new segment), an observation without candidates, or a
step where no state is reachable from any previous state ends the current
chain; decoding restarts from the emissions alone. Breaks are counted and
reported so disconnected predictions are never hidden.
"""
from __future__ import annotations

import heapq
from dataclasses import asdict, dataclass

import numpy as np

from . import config
from .network import RoadNetwork


@dataclass
class HMMParams:
    beta_m: float = config.HMM_DEFAULTS["beta_m"]
    emission_weight: float = config.HMM_DEFAULTS["emission_weight"]
    max_detour_m: float = config.HMM_DEFAULTS["max_detour_m"]
    min_prob: float = config.HMM_DEFAULTS["min_prob"]

    @classmethod
    def from_dict(cls, d: dict) -> "HMMParams":
        return cls(**{k: float(d[k]) for k in cls.__dataclass_fields__ if k in d})

    def to_dict(self) -> dict:
        return asdict(self)


class RouteDistance:
    """Shortest along-network distance between positions on two arcs."""

    def __init__(self, net: RoadNetwork):
        self.length = net.length
        self.v = net.v
        self.order, self.starts = net.successors_csr()

    def successors(self, a: int) -> np.ndarray:
        s = self.v[a]
        return self.order[self.starts[s]:self.starts[s + 1]]

    def from_position(self, a: int, ta: float, targets: dict, limit: float) -> dict:
        """Distances from (arc a, fraction ta) to every target {arc: [(col, tb), ...]} within limit."""
        out = {}
        L = self.length
        if a in targets:  # staying on the same arc (small backward moves = GPS noise)
            for col, tb in targets[a]:
                out[col] = abs(tb - ta) * L[a]
        remaining = set(targets) - {a}
        if not remaining:
            return out
        rem = (1.0 - ta) * L[a]
        if rem > limit:
            return out
        heap = [(rem, int(b)) for b in self.successors(a)]
        heapq.heapify(heap)
        seen = set()
        while heap:
            c, b = heapq.heappop(heap)
            if c > limit:
                break
            if b in seen:
                continue
            seen.add(b)
            if b in remaining:
                for col, tb in targets[b]:
                    d = c + tb * L[b]
                    if d < out.get(col, np.inf):
                        out[col] = d
                remaining.discard(b)
                if not remaining:
                    break
            nc = c + L[b]
            if nc <= limit:
                for s in self.successors(b):
                    s = int(s)
                    if s not in seen:
                        heapq.heappush(heap, (nc, s))
        return out


@dataclass
class DecodeInput:
    """Per-observation candidates in time order + precomputed step distances."""
    arcs: list          # list of int arrays (may be empty)
    t_along: list       # list of float arrays
    proba: list         # list of float arrays
    straight: np.ndarray            # straight-line distance to previous fix (m)
    route_dist: list    # per t: (K_{t-1} x K_t) array of route distances, or None at a break


def precompute(arcs, t_along, proba, xy, segment_id, net: RoadNetwork,
               max_detour_m: float = config.HMM_DEFAULTS["max_detour_m"]) -> DecodeInput:
    rd = RouteDistance(net)
    T = len(arcs)
    straight = np.r_[0.0, np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))]
    dists = [None] * T
    for t in range(1, T):
        if segment_id[t] != segment_id[t - 1] or len(arcs[t]) == 0 or len(arcs[t - 1]) == 0:
            continue
        targets = {}
        for col, (b, tb) in enumerate(zip(arcs[t].tolist(), t_along[t].tolist())):
            targets.setdefault(b, []).append((col, tb))
        limit = straight[t] + max_detour_m
        D = np.full((len(arcs[t - 1]), len(arcs[t])), np.inf)
        for i, (a, ta) in enumerate(zip(arcs[t - 1].tolist(), t_along[t - 1].tolist())):
            for col, d in rd.from_position(a, ta, targets, limit).items():
                D[i, col] = d
        dists[t] = D
    return DecodeInput(arcs, t_along, proba, straight, dists)


def viterbi(inp: DecodeInput, params: HMMParams):
    """Returns (state index per observation or -1, number of chain breaks)."""
    T = len(inp.arcs)
    scores, backs = [None] * T, [None] * T
    breaks = 0
    prev = None
    for t in range(T):
        if len(inp.arcs[t]) == 0:
            prev = None
            continue
        e = params.emission_weight * np.log(np.clip(inp.proba[t], params.min_prob, 1.0))
        D = inp.route_dist[t]
        if prev is None or D is None:
            score, back = e, None
        else:
            with np.errstate(invalid="ignore"):
                logA = -np.abs(D - inp.straight[t]) / params.beta_m   # inf distance -> -inf
            cand = prev[:, None] + logA
            best_i = np.argmax(cand, axis=0)
            best = cand[best_i, np.arange(cand.shape[1])]
            if not np.isfinite(best).any():
                breaks += 1
                score, back = e, None
            else:
                score, back = best + e, best_i
        scores[t], backs[t] = score, back
        prev = score

    state = np.full(T, -1, dtype=np.int64)
    nxt = None  # state chosen at t+1 within the same chain
    for t in range(T - 1, -1, -1):
        if scores[t] is None:
            nxt = None
            continue
        chain_end = (t == T - 1) or scores[t + 1] is None or backs[t + 1] is None
        if chain_end or nxt is None:
            j = int(np.argmax(scores[t]))
        else:
            j = int(backs[t + 1][nxt])
        state[t] = j
        nxt = j
    return state, breaks


def decode_arcs(inp: DecodeInput, params: HMMParams):
    """Arc id per observation (-1 if no candidate) for HMM and classifier-only."""
    state, breaks = viterbi(inp, params)
    hmm = np.array([inp.arcs[t][s] if s >= 0 else -1 for t, s in enumerate(state)], dtype=np.int64)
    clf = np.array([a[int(np.argmax(p))] if len(a) else -1 for a, p in zip(inp.arcs, inp.proba)],
                   dtype=np.int64)
    return hmm, clf, breaks
