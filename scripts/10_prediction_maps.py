"""Folium maps comparing GPS, reference route, classifier-only route and HMM route.

Reads results/predictions/<trip>.csv written by scripts/09_hmm_decode.py and
reuses the drawing helpers of scripts/03_folium_maps.py. Layers can be
toggled so the reference and both predictions can be compared one by one.

Output: maps/prediction_<trip>.html (default: every student-test trip)

Usage:  python scripts/10_prediction_maps.py [--trips 00000052 00000091]
"""
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import folium
import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.geo import LocalProjection
from pathsnap.io import load_trip
from pathsnap.network import RoadNetwork

_spec = importlib.util.spec_from_file_location("eda_maps", ROOT / "scripts" / "03_folium_maps.py")
eda_maps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eda_maps)

COLORS = {**eda_maps.COLORS, "clf": "#f28e2b", "hmm": "#d62728"}


def collapse(seq):
    seq = seq[seq >= 0]
    return seq[np.r_[True, seq[1:] != seq[:-1]]] if len(seq) else seq


def make(trip_id):
    trip = load_trip(config.DATA_DIR / "student_test" / trip_id, "student_test")
    pred = pd.read_csv(config.RESULTS_DIR / "predictions" / f"{trip_id}.csv")
    lon, lat = trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy()
    proj = LocalProjection.from_points(lon, lat)
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    x, y = proj.to_xy(lon, lat)

    m = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], zoom_start=14,
                   tiles="cartodbpositron", control_scale=True)
    fg = folium.FeatureGroup(name="Road network (nearby arcs)", show=True)
    folium.GeoJson(eda_maps.arcs_geojson(trip, eda_maps.arcs_near_track(net, x, y, eda_maps.NETWORK_PAD_M)),
                   style_function=lambda _: {"color": COLORS["network"], "weight": 1.5, "opacity": 0.6}).add_to(fg)
    fg.add_to(m)

    layers = [("Reference route", trip.route, COLORS["route"], 8, 0.45, True),
              ("Classifier-only route", collapse(pred["arc_classifier"].to_numpy()), COLORS["clf"], 3, 0.9, False),
              ("Classifier + HMM route", collapse(pred["arc_hmm"].to_numpy()), COLORS["hmm"], 3.5, 0.95, True)]
    for name, seq, color, weight, opacity, show in layers:
        g = folium.FeatureGroup(name=name, show=show)
        for run in eda_maps.route_polylines(trip, seq):
            folium.PolyLine(run, color=color, weight=weight, opacity=opacity, tooltip=name).add_to(g)
        g.add_to(m)

    g = folium.FeatureGroup(name="GPS observations", show=True)
    step = max(1, len(lon) // eda_maps.MAX_GPS_MARKERS)
    for i in range(0, len(lon), step):
        folium.CircleMarker([float(lat[i]), float(lon[i])], radius=2.2, color=COLORS["gps"], fill=True,
                            fill_opacity=0.7, weight=0, tooltip=f"obs {i}").add_to(g)
    g.add_to(m)

    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(eda_maps.legend_html([
        ("Road network", COLORS["network"]), ("GPS observations", COLORS["gps"]),
        ("Reference route", COLORS["route"]), ("Classifier-only route", COLORS["clf"]),
        ("Classifier + HMM route", COLORS["hmm"])])))
    config.MAPS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.MAPS_DIR / f"prediction_{trip_id}.html"
    m.save(str(out))
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="+")
    a = ap.parse_args()
    trips = a.trips or sorted(p.stem for p in (config.RESULTS_DIR / "predictions").glob("*.csv"))
    for t in trips:
        make(t)
