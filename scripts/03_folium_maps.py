"""Interactive Folium maps of a trip: GPS fixes, road network, reference route
and (optionally) a predicted route.

Layers are toggleable (LayerControl) and a legend is drawn in the corner.
Only arcs near the trajectory are drawn so large networks stay responsive.

Usage examples
  python scripts/03_folium_maps.py                                   # default EDA trips
  python scripts/03_folium_maps.py --partition train --trip 00000046
  python scripts/03_folium_maps.py --partition student_test --trip 00000052 \
         --pred-csv results/predictions/00000052.csv   # columns: obs_idx, arc_id

Output: maps/<partition>_<trip>.html
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import folium
import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.geo import LocalProjection
from pathsnap.io import load_trip
from pathsnap.network import RoadNetwork

DEFAULT_TRIPS = [("train", "00000005"), ("train", "00000046"), ("student_test", "00000052")]
MAX_NETWORK_ARCS = 25_000
MAX_GPS_MARKERS = 3_000
NETWORK_PAD_M = 250

COLORS = {"network": "#9a9a9a", "gps": "#1f4fd1", "route": "#2a9d3a", "pred": "#d62728"}


def arcs_near_track(net: RoadNetwork, x, y, pad):
    box = (x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad)
    inside = ((np.maximum(net.ax, net.bx) >= box[0]) & (np.minimum(net.ax, net.bx) <= box[1]) &
              (np.maximum(net.ay, net.by) >= box[2]) & (np.minimum(net.ay, net.by) <= box[3]))
    idx = np.flatnonzero(inside)
    if len(idx) > MAX_NETWORK_ARCS:  # keep the arcs closest to the track centre
        cx, cy = np.median(x), np.median(y)
        d = np.hypot((net.ax[idx] + net.bx[idx]) / 2 - cx, (net.ay[idx] + net.by[idx]) / 2 - cy)
        idx = idx[np.argsort(d)[:MAX_NETWORK_ARCS]]
    return idx


def arcs_geojson(trip, arc_ids):
    """One MultiLineString feature (GeoJSON is lon, lat order)."""
    lon = trip.nodes["lon"].to_numpy()
    lat = trip.nodes["lat"].to_numpy()
    u = trip.arcs["u"].to_numpy()[arc_ids]
    v = trip.arcs["v"].to_numpy()[arc_ids]
    # two-way streets appear twice; draw each undirected segment once
    key = np.unique(np.sort(np.c_[u, v], axis=1), axis=0)
    lines = [[[float(lon[a]), float(lat[a])], [float(lon[b]), float(lat[b])]] for a, b in key]
    return {"type": "Feature", "properties": {},
            "geometry": {"type": "MultiLineString", "coordinates": lines}}


def route_polylines(trip, arc_seq):
    """Split an arc sequence into connected runs -> list of [(lat, lon), ...]."""
    lon = trip.nodes["lon"].to_numpy()
    lat = trip.nodes["lat"].to_numpy()
    u = trip.arcs["u"].to_numpy()
    v = trip.arcs["v"].to_numpy()
    runs, cur = [], []
    for k, a in enumerate(arc_seq):
        if cur and u[a] != v[arc_seq[k - 1]]:
            runs.append(cur)
            cur = []
        if not cur:
            cur.append((float(lat[u[a]]), float(lon[u[a]])))
        cur.append((float(lat[v[a]]), float(lon[v[a]])))
    if cur:
        runs.append(cur)
    return runs


def legend_html(entries):
    items = "".join(
        f'<div><span style="display:inline-block;width:18px;height:4px;background:{c};'
        f'margin-right:6px;vertical-align:middle"></span>{name}</div>' for name, c in entries)
    return ('<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;'
            'padding:8px 12px;border:1px solid #888;border-radius:4px;font-size:13px">'
            f'<b>Legend</b>{items}</div>')


def make_map(partition, trip_id, pred_csv=None):
    trip = load_trip(config.DATA_DIR / partition / trip_id, partition)
    lon = trip.track["lon"].to_numpy()
    lat = trip.track["lat"].to_numpy()
    proj = LocalProjection.from_points(lon, lat)
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    x, y = proj.to_xy(lon, lat)

    m = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], zoom_start=14,
                   tiles="cartodbpositron", control_scale=True)

    fg_net = folium.FeatureGroup(name="Road network (nearby arcs)", show=True)
    folium.GeoJson(arcs_geojson(trip, arcs_near_track(net, x, y, NETWORK_PAD_M)),
                   style_function=lambda _: {"color": COLORS["network"], "weight": 1.5, "opacity": 0.7}
                   ).add_to(fg_net)
    fg_net.add_to(m)

    legend = [("Road network", COLORS["network"]), ("GPS observations", COLORS["gps"])]
    if trip.route is not None:
        fg_route = folium.FeatureGroup(name="Reference route", show=True)
        for run in route_polylines(trip, trip.route):
            folium.PolyLine(run, color=COLORS["route"], weight=6, opacity=0.65,
                            tooltip="reference route").add_to(fg_route)
        fg_route.add_to(m)
        legend.append(("Reference route", COLORS["route"]))

    if pred_csv:
        pred = pd.read_csv(pred_csv)
        seq = pred["arc_id"].to_numpy()
        seq = seq[np.r_[True, seq[1:] != seq[:-1]]]  # collapse repeats
        fg_pred = folium.FeatureGroup(name="Predicted route", show=True)
        for run in route_polylines(trip, seq):
            folium.PolyLine(run, color=COLORS["pred"], weight=3, opacity=0.9,
                            tooltip="predicted route").add_to(fg_pred)
        fg_pred.add_to(m)
        legend.append(("Predicted route", COLORS["pred"]))

    fg_gps = folium.FeatureGroup(name="GPS observations", show=True)
    step = max(1, len(lon) // MAX_GPS_MARKERS)
    t = trip.track["t"].to_numpy()
    for i in range(0, len(lon), step):
        folium.CircleMarker([float(lat[i]), float(lon[i])], radius=2.5, color=COLORS["gps"],
                            fill=True, fill_opacity=0.8, weight=0,
                            tooltip=f"obs {i}  t={t[i]:.0f}s").add_to(fg_gps)
    fg_gps.add_to(m)
    folium.Marker([float(lat[0]), float(lon[0])], tooltip="start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([float(lat[-1]), float(lon[-1])], tooltip="end",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)

    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(legend_html(legend)))

    config.MAPS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.MAPS_DIR / f"{partition}_{trip_id}.html"
    m.save(str(out))
    print("saved", out, f"(GPS markers every {step} fix(es))")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition")
    ap.add_argument("--trip")
    ap.add_argument("--pred-csv")
    a = ap.parse_args()
    if a.trip:
        make_map(a.partition or "train", a.trip, a.pred_csv)
    else:
        for part, tid in DEFAULT_TRIPS:
            make_map(part, tid)
