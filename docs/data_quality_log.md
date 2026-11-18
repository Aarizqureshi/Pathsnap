# Data-quality log, assumptions and decisions

Partitions analysed: **train (80 trips)** and **student_test (10 trips)**.
The evaluator-test partition was not available to us and was not used.
Every number below is produced by the scripts named in brackets and can be
regenerated from the repository.

## 1. What the raw data looks like  [01_dataset_inventory.py]

| Check | train | student_test |
|---|---|---|
| Trips / GPS observations | 80 / 208,417 | 10 / 19,160 |
| Missing files | 0 | 0 |
| NaN values, out-of-range coordinates | 0, 0 | 0, 0 |
| Duplicate timestamps, out-of-order timestamps | 0, 0 | 0, 0 |
| Trips with at least one gap > 10 s | 40 | 2 |
| Raw steps implying > 70 m/s | 147 | 1 |
| Self-loop arcs / duplicate arcs / zero-length arcs | 22 / 1,775 / 100 | 2 / 128 / 6 |
| Trips whose network has > 1 weakly connected component | 79 | 10 |
| Trips whose reference route has a break (consecutive arcs not sharing a node) | 10 | 1 |

* Sampling is nominally 1 Hz: 96.7 % of all steps are exactly 1 s. Gaps range
  up to 153,064 s (trip 00000090).
* Trips are located on five continents, so a single projected CRS is not
  appropriate. We use a per-trip local tangent-plane projection
  (`geo.LocalProjection`). Max |projected - haversine| over every consecutive
  pair of fixes in both partitions: **0.0023 m**.
* Road networks are per trip and vary from 649 to 307,392 directed arcs.
  Two-way streets are stored as two arcs with opposite direction and
  identical geometry.
* Networks are fragmented (median 19 components per trip), but the largest
  component holds a median 98 % of nodes. This matters for the HMM: transitions
  between components must be treated as impossible or as a reset.

## 2. Cleaning decisions  [04_preprocess_and_label.py]

| Step | Rule | Effect (train) |
|---|---|---|
| Invalid coordinates | drop non-finite / out-of-range | 0 dropped |
| Ordering | stable sort by time, drop duplicate timestamps | 0 dropped (already clean) |
| Spike removal | drop a fix only if speed in AND out > 70 m/s while skipping it is plausible | 0 dropped |
| Segmentation | new segment when dt > 10 s | 590 gaps -> segments |
| Kinematics | heading from fixes i-2 to i+2 (same segment); invalid if displacement < 3 m | 98 % of fixes have a valid heading |

**Why the spike filter removes nothing although 147 steps exceed 70 m/s:**
95 of those steps are consecutive in trip 00000013 (a sustained ~86 m/s run)
and 40 are in trip 00000032 (~73 m/s). These are consistent high-speed
movement, not jumps. The rest are one-off position shifts where the following
fixes stay shifted. We deliberately keep both, because deleting real
observations would remove points we are required to predict. The classifier
and HMM are expected to absorb this noise.

## 3. Turning the reference route into labels  [labels.py]

The `.route` file is an ordered arc list with no link to individual
observations. We assign every observation a position on the route using a
**monotone dynamic-programming alignment**. Positions may only move forward,
and the total GPS-to-arc distance is minimised. This avoids matching a fix to
a part of the route the vehicle has already left (loops, U-turns, opposite
carriageways).

* Median GPS-to-aligned-arc distance: 4.63 m (train), 4.74 m (student_test).
* **Observations further than 40 m from their aligned arc get
  `label_valid = 0`** and are excluded from training and scoring:
  11.65 % of train observations, 0.21 % of student_test observations.

Two trips explain most of the invalid train labels:

* **00000054**: the reference route (31 km) ends after ~50 min, but the logger
  keeps recording for another ~4.5 h at ~2 m/s, about 90 m from any road
  (169 km of track). Only 9.9 % of its observations have a valid label.
* **00000033**: the GPS stays on roads (median 6 m from the nearest arc), but
  the reference route covers only 6.4 km of a 47.8 km track. 33 % of its
  observations have a valid label.

## 4. Limitations of the reference labels (do not over-trust them)

* The route is a reference, not verified ground truth. 11 trips contain at
  least one break in route continuity.
* 10 train trips revisit the same arc. The monotone alignment handles this,
  but the timing of the second visit is inferred.
* Duplicate arcs (same from/to nodes, different ids) exist in the networks.
  If the route uses one id, the identical twin counts as a negative.
* Near arc boundaries, the "true" arc of a fix is ambiguous within the
  GPS noise (~5 m), so some label noise at segment transitions is unavoidable.
* The 40 m validity threshold was chosen from the train distance histogram
  (`results/figures/preprocessing/align_distance_hist_train.png`), not from
  student_test results.

## 5. Candidate generation and features  [05_build_features.py, 06_feature_analysis.py]

* Candidates: arcs within **50 m**, nearest first, at most **12** per
  observation (fallback: up to 200 m if none within 50 m).
* Chosen from the train recall sweep. 50 m / K = 12 gives 99.73 % recall of
  the reference arc. K = 16 adds only 0.1 pp for ~30 % more rows.
  Note that recall cannot grow beyond 40 m by construction, because
  labels further away are already marked invalid.
* Candidate miss rate: 0.27 % (train), 0.06 % (student_test). Mean candidates
  per observation: 7.5. Class ratio: ~6 negatives per positive.
* Rule baselines (top-1 on valid observations, student_test):
  nearest arc 54.2 %, heading-aware nearest arc 84.9 %. Models must beat
  84.9 % to be useful.
* Observation-level features (speed, turn rate, dt, heading validity) have
  near-zero univariate separability. That is expected, because they are
  identical for every candidate of the same observation. They can only help
  through interactions, so the models will show whether they are worth
  keeping.
