"""Central configuration for PathSnap.

Every tunable constant used by preprocessing, label alignment, candidate
generation and feature engineering lives here so that experiments are
reproducible and the values can be quoted directly in the report.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[2]

# Raw data root. Expected layout:
#   <DATA_DIR>/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
#   <DATA_DIR>/student_test/<trip_id>/...
# Override with the PATHSNAP_DATA environment variable.
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", REPO_ROOT / "data" / "raw"))
PROCESSED_DIR = Path(os.environ.get("PATHSNAP_PROCESSED", REPO_ROOT / "data" / "processed"))
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MAPS_DIR = REPO_ROOT / "maps"
MODELS_DIR = REPO_ROOT / "models"
CONFIGS_DIR = REPO_ROOT / "configs"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
HMM_CONFIG_PATH = CONFIGS_DIR / "hmm.json"

PARTITIONS = ("train", "student_test")

RANDOM_SEED = 42

# -------------------------------------------------------- preprocessing
# Plausibility box for coordinates (degrees).
LON_RANGE = (-180.0, 180.0)
LAT_RANGE = (-90.0, 90.0)

# Implied speeds above this between consecutive fixes are treated as GPS
# jumps (~252 km/h, above any legal road speed in the dataset's countries).
MAX_PLAUSIBLE_SPEED_MS = 70.0

# A time step larger than this starts a new continuous track segment.
# The data is nominally 1 Hz; 10 s still lets a car travel ~150 m at
# city speed, beyond which sequential smoothness is not trustworthy.
GAP_THRESHOLD_S = 10.0

# Heading is estimated from the displacement between the fixes
# HEADING_HALF_WINDOW steps before and after a point (same segment only).
HEADING_HALF_WINDOW = 2
# Below this displacement (metres) over the window the heading is noise.
MIN_DISPLACEMENT_FOR_HEADING_M = 3.0

# ------------------------------------------------------- label alignment
# Observations further than this from their aligned reference arc get
# label_valid = 0 and are excluded from training and scoring (e.g. trip
# 00000054 keeps logging for hours after the reference route ends).
LABEL_MAX_ALIGN_DIST_M = 40.0

# ------------------------------------------------- candidate generation
# Chosen from the train-set GPS-to-reference-arc distance distribution
# (see results/features/candidate_recall.csv).
CANDIDATE_RADIUS_M = 50.0
CANDIDATE_MAX_K = 12
# Fallback radius for observations with no candidate inside the main radius.
CANDIDATE_FALLBACK_RADIUS_M = 200.0
GRID_CELL_M = 100.0  # must be >= every query radius divided by the ring size

# ---------------------------------------------------- feature engineering
CONTEXT_HALF_WINDOW = 3  # neighbours each side used for temporal-support features

FEATURE_COLUMNS = [
    "dist_m",
    "dist_rank",
    "dist_minus_best_m",
    "n_candidates",
    "heading_diff_deg",
    "heading_valid",
    "speed_ms",
    "turn_rate_dps",
    "arc_length_m",
    "off_segment",
    "prev_dist_m",
    "next_dist_m",
    "context_mean_dist_m",
    "end_out_degree",
    "dt_prev_s",
]

# ------------------------------------------------------------- modelling
CV_FOLDS = 5                 # GroupKFold by trip on the train partition
# Share of train observations (all candidates of each sampled observation)
# used for cross-validation, so all five models see the same rows and the
# comparison finishes on a laptop. Final fits use all valid train rows.
CV_OBS_FRAC = 0.25
# Instance/kernel methods scale badly with rows; their training sets are capped.
MAX_TRAIN_ROWS = {"knn": 200_000, "svm_rbf": 20_000}

# Bayesian optimisation (Optuna TPE)
TUNE_OBS_FRAC = 0.10
TUNE_FOLDS = 3
TUNE_TRIALS = 25

# ------------------------------------------------------------- HMM / Viterbi
HMM_DEFAULTS = {
    "beta_m": 10.0,          # scale of |route distance - straight-line distance| (Newson & Krumm)
    "emission_weight": 1.0,  # weight of log classifier probability vs transition
    "max_detour_m": 150.0,   # route search stops this far beyond the straight-line distance
    "min_prob": 1e-6,
}
HMM_GRID = {"beta_m": [3.0, 5.0, 10.0, 20.0, 40.0], "emission_weight": [0.5, 1.0, 2.0]}
HMM_TUNE_HOLDOUT_TRIPS = 16  # train trips held out to tune the decoder
