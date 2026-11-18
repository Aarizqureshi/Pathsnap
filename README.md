# PathSnap: Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University (RR Campus).

PathSnap matches noisy GPS observations to the road network they were
recorded on. The pipeline works in four stages:

```
raw trip files ─► cleaning & labels ─► candidate arcs + features ─► classifier ─► HMM / Viterbi ─► route
 (.track .nodes     (preprocess.py,        (candidates.py,           (phase 2)      (phase 3)
  .arcs .route)      labels.py)             features.py)
```

1. **Preprocessing.** Each GPS track is validated, sorted, projected to metres
   with a per-trip local projection, split at time gaps, and given speed,
   heading and turn rate.
2. **Labels.** The reference route (an ordered arc list) is aligned to the
   observations with a monotone dynamic-programming alignment. Observations
   more than 40 m from their aligned arc are marked `label_valid = 0`.
3. **Candidates and features.** For every observation, the arcs within 50 m
   (at most 12) become candidates. Each (observation, candidate) pair gets
   15 features: geometric, directional, temporal-support and context.
4. **Models (phase 2).** At least five classifiers score candidates, and
   Bayesian optimisation tunes the best one.
5. **HMM/Viterbi (phase 3).** Classifier probabilities become emission scores
   and network connectivity defines the transitions.
6. **Deployment (phase 4).** A FastAPI backend (laptop 1) and an evaluator UI
   (laptop 2).

## Repository layout

```
src/pathsnap/        library code
  config.py          every tunable constant (radii, thresholds, seeds)
  io.py              raw file readers
  geo.py             projection, haversine, point-to-segment geometry
  network.py         road network, connectivity, grid spatial index
  preprocess.py      track cleaning + kinematics
  labels.py          route-to-observation alignment
  pipeline.py        raw trip -> cleaned, labelled observations
  candidates.py      candidate arc generation
  features.py        per-candidate features
scripts/             runnable steps, numbered in execution order
results/             CSV outputs and figures (committed)
maps/                Folium HTML maps
docs/                data-quality log, assumptions, decisions
data/raw/            raw data (NOT committed)
data/processed/      generated tables (NOT committed)
```

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Put the provided data under `data/raw/` so it looks like this:

```
data/raw/train/00000001/00000001.track ...
data/raw/student_test/00000000/00000000.track ...
```

Alternatively, point the scripts to the data with an environment variable,
e.g. `set PATHSNAP_DATA=C:\path\to\Student folder` on Windows CMD.

## Running (phase 1: EDA, preprocessing, features)

Run from the repository root, in this order:

| Step | Command | Produces |
|---|---|---|
| 1 | `python scripts/01_dataset_inventory.py` | `results/eda/*.csv` (inventory, missing values, projection check) |
| 2 | `python scripts/02_eda_plots.py` | `results/figures/eda/*.png` |
| 3 | `python scripts/03_folium_maps.py` | `maps/*.html` (GPS, network, reference route, layer control) |
| 4 | `python scripts/04_preprocess_and_label.py` | `data/processed/points_*.csv.gz`, `results/preprocessing/*` |
| 5 | `python scripts/05_build_features.py` | `data/processed/features_*.csv.gz`, `results/features/*` |
| 6 | `python scripts/06_feature_analysis.py` | feature statistics, correlation, rule baselines |

Steps 4–6 take about 2 minutes in total on a laptop.

## Phase-1 results

| | train | student_test |
|---|---|---|
| Trips / observations | 80 / 208,417 | 10 / 19,160 |
| Observations with a valid label | 88.4 % | 99.8 % |
| Candidate miss rate (reference arc not in candidate set) | 0.27 % | 0.06 % |
| Mean candidates per observation | 7.5 | 7.7 |
| Top-1, nearest-arc rule | 56.1 % | 54.2 % |
| Top-1, heading-aware nearest-arc rule | 85.3 % | 84.9 % |

See `docs/data_quality_log.md` for every data issue found and how it was handled.

## Leakage rules we follow

* Only the 80 training trips are used for fitting, cross-validation and tuning.
  Cross-validation is grouped by trip.
* The 10 student-test trips are used only for development comparisons.
* Evaluator-test labels never enter the prediction pipeline or the UI.
  Nothing is hardcoded per trip.
* Preprocessing is label-free and identical for every partition. Learned
  transforms (e.g. scalers) are fitted on train only.

## Team

| GitHub | Phase-1 responsibility |
|---|---|
| Aarizqureshi | Repository setup, configuration, data I/O, geodesy |
| aarohijaiswal | Road network & connectivity, dataset inventory, EDA plots, Folium maps |
| nramsaigovind-rgb | Track cleaning, route-label alignment, preprocessing pipeline |
| SharGIT-prog | Candidate generation, feature engineering, feature analysis |

## AI assistance disclosure

Parts of this codebase were developed with the help of an AI assistant
(Claude, Anthropic). All experiments were run by the team, and all reported
numbers come from the scripts in this repository. Every member has reviewed
and can explain the parts they committed.
