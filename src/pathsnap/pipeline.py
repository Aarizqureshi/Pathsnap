"""One function that takes a raw trip to cleaned, labelled observations.

Shared by the offline scripts and (later) the API so that training and
inference apply exactly the same preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .geo import LocalProjection
from .io import Trip
from .labels import align_route, label_quality
from .network import RoadNetwork, route_continuity
from .preprocess import clean_track


@dataclass
class PreparedTrip:
    trip: Trip
    proj: LocalProjection
    net: RoadNetwork
    points: pd.DataFrame          # cleaned observations (+ labels if a route exists)
    log: dict = field(default_factory=dict)

    @property
    def xy(self):
        return self.points["x"].to_numpy(), self.points["y"].to_numpy()


def prepare_trip(trip: Trip, with_labels: bool = True) -> PreparedTrip:
    proj = LocalProjection.from_points(trip.track["lon"], trip.track["lat"])
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    points, log = clean_track(trip.track, proj)
    log = {"trip_id": trip.trip_id, "partition": trip.partition, **log}

    if with_labels and trip.route is not None and len(points):
        x, y = points["x"].to_numpy(), points["y"].to_numpy()
        pos, true_arc, adist = align_route(x, y, trip.route, net)
        points["route_pos"] = pos
        points["true_arc"] = true_arc
        points["align_dist_m"] = adist
        points["label_valid"] = (adist <= config.LABEL_MAX_ALIGN_DIST_M).astype(np.int8)
        log.update(label_quality(adist, pos, len(trip.route)))
        log["label_valid_frac"] = float(points["label_valid"].mean())
        log["route_len_arcs"] = int(len(trip.route))
        log["route_continuity"] = route_continuity(trip.route, net)
    return PreparedTrip(trip, proj, net, points, log)
