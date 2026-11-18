"""Readers for the raw PathSnap trip files.

Each trip directory ``<trip_id>/`` contains four header-less,
whitespace-separated text files:

* ``.track``  longitude, latitude, timestamp (s)    -> GPS observations
* ``.nodes``  longitude, latitude                    -> road nodes (row index = node id)
* ``.arcs``   from_node_id, to_node_id               -> directed road segments (row index = arc id)
* ``.route``  arc_id per line                        -> reference route, in travel order

A row of the track file does NOT correspond to a row of any network file;
the link between observations and arcs has to be reconstructed
(see :mod:`pathsnap.labels`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from . import config

FILE_TYPES = ("track", "nodes", "arcs", "route")


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame          # columns: lon, lat, t
    nodes: pd.DataFrame          # columns: lon, lat
    arcs: pd.DataFrame           # columns: u, v
    route: Optional[np.ndarray]  # arc ids in travel order, None if not supplied
    files: dict = field(default_factory=dict)  # file type -> Path

    @property
    def n_obs(self) -> int:
        return len(self.track)


def _read_numeric(path: Path, n_cols: int, dtype) -> np.ndarray:
    """Read a whitespace separated, header-less numeric file into (n, n_cols)."""
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, n_cols), dtype=dtype)
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if arr.shape[1] != n_cols:
        raise ValueError(f"{path.name}: expected {n_cols} columns, found {arr.shape[1]}")
    return arr


def trip_files(trip_dir: Path) -> dict:
    trip_id = trip_dir.name
    return {ft: trip_dir / f"{trip_id}.{ft}" for ft in FILE_TYPES}


def load_trip(trip_dir: Path | str, partition: str = "unknown") -> Trip:
    trip_dir = Path(trip_dir)
    files = trip_files(trip_dir)

    track = _read_numeric(files["track"], 3, np.float64)
    nodes = _read_numeric(files["nodes"], 2, np.float64)
    arcs = _read_numeric(files["arcs"], 2, np.int64)
    route_arr = _read_numeric(files["route"], 1, np.int64)

    return Trip(
        trip_id=trip_dir.name,
        partition=partition,
        track=pd.DataFrame(track, columns=["lon", "lat", "t"]),
        nodes=pd.DataFrame(nodes, columns=["lon", "lat"]),
        arcs=pd.DataFrame(arcs, columns=["u", "v"]),
        route=route_arr[:, 0] if len(route_arr) else None,
        files=files,
    )


def list_trip_dirs(partition: str, data_dir: Path | str | None = None) -> list[Path]:
    root = Path(data_dir or config.DATA_DIR) / partition
    if not root.exists():
        raise FileNotFoundError(
            f"Partition folder not found: {root}. Put the raw data under "
            f"{config.DATA_DIR} or set PATHSNAP_DATA."
        )
    return sorted(p for p in root.iterdir() if p.is_dir())


def iter_trips(partition: str, data_dir: Path | str | None = None) -> Iterator[Trip]:
    for d in list_trip_dirs(partition, data_dir):
        yield load_trip(d, partition)
