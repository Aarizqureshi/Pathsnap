"""Road-network representation, connectivity analysis and spatial index.

Arcs are *directed*: a two-way street appears as two arcs (u->v and v->u)
with identical geometry but opposite bearings. Arc ids are row indices of
the ``.arcs`` file and must be preserved end-to-end because the reference
route and the final predictions are expressed in arc ids.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .geo import LocalProjection, bearing_deg, project_to_segments


class RoadNetwork:
    def __init__(self, nodes: pd.DataFrame, arcs: pd.DataFrame, proj: LocalProjection):
        self.proj = proj
        self.n_nodes = len(nodes)
        self.n_arcs = len(arcs)
        self.node_x, self.node_y = proj.to_xy(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
        self.u = arcs["u"].to_numpy(np.int64)
        self.v = arcs["v"].to_numpy(np.int64)
        self.ax, self.ay = self.node_x[self.u], self.node_y[self.u]
        self.bx, self.by = self.node_x[self.v], self.node_y[self.v]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = bearing_deg(self.bx - self.ax, self.by - self.ay)
        self.out_degree = np.bincount(self.u, minlength=self.n_nodes)
        self.in_degree = np.bincount(self.v, minlength=self.n_nodes)
        self._grid = None

    # ------------------------------------------------------------ topology
    def successors_csr(self):
        """CSR adjacency arc -> arcs that start where it ends (used by the HMM)."""
        order = np.argsort(self.u, kind="stable")
        starts = np.searchsorted(self.u[order], np.arange(self.n_nodes + 1))
        return order, starts  # successors of arc a: order[starts[v[a]]:starts[v[a]+1]]

    def weak_components(self) -> np.ndarray:
        """Weakly connected component label per node (union-find, path halving)."""
        parent = np.arange(self.n_nodes)

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a, b in zip(self.u.tolist(), self.v.tolist()):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        return np.array([find(i) for i in range(self.n_nodes)])

    def connectivity_summary(self) -> dict:
        comp = self.weak_components()
        degree = self.out_degree + self.in_degree
        _, sizes = np.unique(comp, return_counts=True)
        pairs = self.u * (self.n_nodes + 1) + self.v
        return {
            "n_nodes": self.n_nodes,
            "n_arcs": self.n_arcs,
            "n_components": int(len(sizes)),
            "largest_component_frac": float(sizes.max() / self.n_nodes) if self.n_nodes else np.nan,
            "isolated_nodes": int((degree == 0).sum()),
            "self_loops": int((self.u == self.v).sum()),
            "duplicate_arcs": int(self.n_arcs - len(np.unique(pairs))),
            "zero_length_arcs": int((self.length < 1e-6).sum()),
            "median_arc_length_m": float(np.median(self.length)) if self.n_arcs else np.nan,
            "max_arc_length_m": float(self.length.max()) if self.n_arcs else np.nan,
        }

    # ------------------------------------------------------- spatial index
    def _build_grid(self, cell: float):
        """Uniform grid: every arc is registered in each cell its bounding box touches."""
        c0x = np.floor(np.minimum(self.ax, self.bx) / cell).astype(np.int64)
        c1x = np.floor(np.maximum(self.ax, self.bx) / cell).astype(np.int64)
        c0y = np.floor(np.minimum(self.ay, self.by) / cell).astype(np.int64)
        c1y = np.floor(np.maximum(self.ay, self.by) / cell).astype(np.int64)
        nx = c1x - c0x + 1
        ny = c1y - c0y + 1
        n_cells = nx * ny
        arc_rep = np.repeat(np.arange(self.n_arcs), n_cells)
        k = np.arange(n_cells.sum()) - np.repeat(np.cumsum(n_cells) - n_cells, n_cells)
        cx = c0x[arc_rep] + k % nx[arc_rep]
        cy = c0y[arc_rep] + k // nx[arc_rep]
        keys = self._cell_key(cx, cy)
        order = np.argsort(keys, kind="stable")
        self._grid = (cell, keys[order], arc_rep[order])

    @staticmethod
    def _cell_key(cx, cy):
        return (cx + 2**20) * 2**21 + (cy + 2**20)

    def query_radius(self, px, py, radius: float):
        """All (point, arc) pairs with point-to-arc distance <= radius.

        Returns arrays ``point_idx, arc_id, dist, t`` (unsorted).
        """
        if self._grid is None:
            self._build_grid(config.GRID_CELL_M)
        cell, keys, arcs = self._grid
        ring = int(np.ceil(radius / cell))
        px = np.asarray(px, float)
        py = np.asarray(py, float)
        pcx = np.floor(px / cell).astype(np.int64)
        pcy = np.floor(py / cell).astype(np.int64)
        offs = np.arange(-ring, ring + 1)
        ox, oy = np.meshgrid(offs, offs)
        ox, oy = ox.ravel(), oy.ravel()
        n_p, n_o = len(px), len(ox)
        q_point = np.repeat(np.arange(n_p), n_o)
        q_key = self._cell_key(np.repeat(pcx, n_o) + np.tile(ox, n_p),
                               np.repeat(pcy, n_o) + np.tile(oy, n_p))
        lo = np.searchsorted(keys, q_key, "left")
        hi = np.searchsorted(keys, q_key, "right")
        cnt = hi - lo
        total = int(cnt.sum())
        if total == 0:
            e = np.empty(0)
            return e.astype(np.int64), e.astype(np.int64), e, e
        point_idx = np.repeat(q_point, cnt)
        pos = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(lo, cnt)
        arc_id = arcs[pos]
        # an arc can sit in several neighbouring cells -> de-duplicate pairs
        pair = np.unique(point_idx * self.n_arcs + arc_id)
        point_idx, arc_id = pair // self.n_arcs, pair % self.n_arcs
        dist, t = project_to_segments(px[point_idx], py[point_idx],
                                      self.ax[arc_id], self.ay[arc_id],
                                      self.bx[arc_id], self.by[arc_id])
        keep = dist <= radius
        return point_idx[keep], arc_id[keep], dist[keep], t[keep]


def route_continuity(route: np.ndarray, net: RoadNetwork) -> float:
    """Fraction of consecutive reference-route arc pairs that share a node."""
    if route is None or len(route) < 2:
        return np.nan
    return float(np.mean(net.v[route[:-1]] == net.u[route[1:]]))


def build_network(trip, proj: LocalProjection | None = None) -> RoadNetwork:
    if proj is None:
        proj = LocalProjection.from_points(trip.track["lon"], trip.track["lat"])
    return RoadNetwork(trip.nodes, trip.arcs, proj)
