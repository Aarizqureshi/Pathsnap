# PathSnap: Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University (RR Campus).

PathSnap matches noisy GPS observations to the road network they were
recorded on. It assigns every observation the directed road segment (arc)
the vehicle was on, then joins those choices into a connected route.

```
raw trip ─► cleaning & labels ─► candidate arcs + features ─► classifier ─► HMM / Viterbi ─► route
             preprocess.py         candidates.py                models.py      hmm.py
             labels.py             features.py                  (5 compared,   inference.py
                                                                 1 tuned)
                                                    FastAPI (laptop 1) ◄──► evaluator UI (laptop 2)
```

1. **Preprocessing.** Each GPS track is validated, sorted, projected to metres
   with a per-trip local projection, split at time gaps, and given speed,
   heading and turn rate.
2. **Labels.** The reference route is aligned to the observations with a
   monotone dynamic-programming alignment. Observations more than 40 m from
   their aligned arc are marked `label_valid = 0`.
3. **Candidates and features.** Arcs within 50 m (at most 12) become
   candidates. Each (observation, candidate) pair gets 15 features.
4. **Models.** Logistic Regression, Gaussian Naive Bayes, KNN, Random Forest
   and RBF SVM are compared with trip-grouped cross-validation. The best is
   tuned with Bayesian optimisation (Optuna TPE).
5. **HMM/Viterbi.** Candidates are the hidden states, classifier
   probabilities are the emissions, and network distance vs straight-line
   distance gives the transitions. The decoder is tuned on held-out train trips.
6. **Deployment.** A FastAPI backend (laptop 1) and a Leaflet evaluator UI
   (laptop 2). See `docs/deployment.md`.

## Repository layout

```
src/pathsnap/        library code
  config.py          every tunable constant (radii, thresholds, seeds, CV, HMM)
  io.py, geo.py      raw file readers, projection and segment geometry
  network.py         road network, connectivity, grid spatial index
  preprocess.py      track cleaning + kinematics
  labels.py          route-to-observation alignment
  pipeline.py        raw trip -> cleaned, labelled observations
  candidates.py      candidate arc generation
  features.py        per-candidate features
  models.py          the five classifiers + data sampling
  metrics.py         candidate, observation and route metrics
  hmm.py             HMM transitions (bounded Dijkstra) + Viterbi
  inference.py       end-to-end prediction for one trip (labels stripped first)
  evaluation.py      server-side scoring against the reference route
scripts/             runnable steps 01-11, in execution order
api/main.py          FastAPI backend
ui/                  evaluator UI (index.html, app.js, style.css)
tests/test_hmm.py    decoder unit tests
models/              final_model.joblib + model_card.json
configs/hmm.json     tuned decoder parameters
results/             CSV outputs, figures, summary.md (committed)
maps/                Folium HTML maps
docs/                data-quality log, deployment guide
data/                raw + processed data (NOT committed)
```

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Point the scripts to the data folder, which must contain `train/` and `student_test/`:

```bat
set PATHSNAP_DATA=C:\path\to\Student folder
```

Or place the data under `data/raw/`.

## Running everything

On Windows, `run_pipeline.bat "C:\path\to\Student folder"` runs all steps in order. Individually:

| Step | Script | Produces |
|---|---|---|
| 1 | `01_dataset_inventory.py` | `results/eda/*.csv` |
| 2 | `02_eda_plots.py` | `results/figures/eda/*.png` |
| 3 | `03_folium_maps.py` | `maps/<partition>_<trip>.html` |
| 4 | `04_preprocess_and_label.py` | cleaning and label logs |
| 5 | `05_build_features.py` | feature tables, candidate recall |
| 6 | `06_feature_analysis.py` | feature separability, rule baselines |
| 7 | `07_compare_models.py` | five-model comparison, CV folds, confusion matrices |
| 8 | `08_bayesian_optimisation.py` | Optuna trials, tuned vs untuned, `models/final_model.joblib` |
| 9 | `09_hmm_decode.py` | decoder grid, `configs/hmm.json`, classifier vs HMM per trip |
| 10 | `10_prediction_maps.py` | `maps/prediction_<trip>.html` |
| 11 | `11_summarise_results.py` | `results/summary.md` (report tables) |

`python tests/test_hmm.py` checks the decoder on a hand-built network. `python tests/test_no_leakage.py` proves predictions are identical with the real route, no route, a random route, or a route that raises if touched.

## Deployment (evaluation day)

* Laptop 1: `start_api.bat "<data folder>"` starts the API on port 8000, offers `student_test` and `evaluator_test` in the UI dropdown and logs every request.
* Laptop 2: `start_ui.bat`, then open `http://localhost:8080` and connect to `http://<laptop-1 IP>:8000`.

Details and a checklist are in `docs/deployment.md`.

## Phase-1 results (data and features)

| | train | student_test |
|---|---|---|
| Trips / observations | 80 / 208,417 | 10 / 19,160 |
| Observations with a valid label | 88.4 % | 99.8 % |
| Candidate miss rate | 0.27 % | 0.06 % |
| Mean candidates per observation | 7.5 | 7.7 |
| Top-1, nearest-arc rule | 56.1 % | 54.2 % |
| Top-1, heading-aware nearest-arc rule | 85.3 % | 84.9 % |

Model, tuning and HMM results are generated into `results/summary.md` by step 11.

## Leakage rules we follow

* Only the 80 training trips are used for fitting, cross-validation and tuning.
  Cross-validation is grouped by trip. The HMM decoder is tuned on held-out
  train trips.
* The 10 student-test trips are used only for development comparisons.
* The prediction path removes the reference route before running.
  Evaluator labels are read only by `evaluation.py`, after prediction.
  The UI receives aggregate metrics only. Nothing is hardcoded per trip.
* Preprocessing is label-free and identical for every partition. Scalers
  live inside the model pipelines and are fitted on training folds only.

## Team

| GitHub | Phase 1 | Phases 2–4 |
|---|---|---|
| Aarizqureshi | Repo setup, config, data I/O, geodesy | Model definitions, metrics, five-model comparison |
| aarohijaiswal | Road network, inventory, EDA plots, Folium maps | Bayesian optimisation, final model, HMM/Viterbi decoder + tests |
| nramsaigovind-rgb | Track cleaning, label alignment, data-quality log | Inference pipeline, server-side evaluation, HMM experiments, prediction maps |
| SharGIT-prog | Candidates, features, feature analysis | FastAPI backend, evaluator UI, deployment scripts, results summary |

## AI assistance disclosure

Parts of this codebase were developed with the help of an AI assistant
(Claude, Anthropic). All experiments were run by the team, and all reported
numbers come from the scripts in this repository. Every member has reviewed
and can explain the parts they committed.
