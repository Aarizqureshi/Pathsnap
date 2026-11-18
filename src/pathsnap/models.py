"""The five candidate classifiers and the data handling shared by every
training script.

Each model scores one (observation, candidate arc) row: P(candidate is the
reference arc). Models from five algorithm families are compared:

    logreg   Logistic Regression          linear / probabilistic (discriminative)
    gnb      Gaussian Naive Bayes         probabilistic (generative)
    knn      Distance-weighted KNN        instance-based
    rf       Random Forest                ensemble of trees (bagging)
    svm_rbf  RBF-kernel SVM (Platt)       kernel-based

Scaling is inside each Pipeline, so it is fitted on the training fold only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config

FEATURES = list(config.FEATURE_COLUMNS)
KEY = ["trip_id", "point_idx"]


@dataclass
class ModelSpec:
    name: str
    family: str
    defaults: dict
    notes: str
    max_train_rows: int | None = None
    search_space: dict = field(default_factory=dict)  # used by Bayesian optimisation


SPECS = {
    "logreg": ModelSpec(
        "logreg", "Linear / probabilistic",
        {"C": 1.0},
        "StandardScaler + L2 logistic regression, class_weight=balanced, lbfgs",
    ),
    "gnb": ModelSpec(
        "gnb", "Probabilistic (generative)",
        {"var_smoothing": 1e-9},
        "Gaussian Naive Bayes, class priors from data, no scaling needed",
    ),
    "knn": ModelSpec(
        "knn", "Instance-based",
        {"n_neighbors": 50, "weights": "distance"},
        "StandardScaler + KNN (euclidean), trained on a capped row sample",
        max_train_rows=config.MAX_TRAIN_ROWS["knn"],
    ),
    "rf": ModelSpec(
        "rf", "Ensemble (bagged trees)",
        {"n_estimators": 150, "max_leaf_nodes": 2048, "min_samples_leaf": 10, "max_features": "sqrt"},
        "Random forest, class_weight=balanced_subsample, n_jobs=-1; max_leaf_nodes bounds tree size so the "
        "deployed model stays small (an unbounded forest on 1.3M rows grows to GBs)",
    ),
    "svm_rbf": ModelSpec(
        "svm_rbf", "Kernel-based",
        {"C": 1.0, "gamma": round(1 / len(config.FEATURE_COLUMNS), 6)},
        "StandardScaler + RBF SVC, Platt (sigmoid) calibration with 3-fold CV, class_weight=balanced, capped rows; "
        "gamma = 1/n_features, equivalent to sklearn's 'scale' on standardised features",
        max_train_rows=config.MAX_TRAIN_ROWS["svm_rbf"],
    ),
}


def make_model(name: str, params: dict | None = None):
    p = {**SPECS[name].defaults, **(params or {})}
    seed = config.RANDOM_SEED
    if name == "logreg":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(C=p["C"], class_weight="balanced", max_iter=2000))])
    if name == "gnb":
        return GaussianNB(var_smoothing=p["var_smoothing"])
    if name == "knn":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=int(p["n_neighbors"]),
                                                      weights=p["weights"], n_jobs=-1))])
    if name == "rf":
        return RandomForestClassifier(n_estimators=int(p["n_estimators"]),
                                      max_leaf_nodes=int(p["max_leaf_nodes"]),
                                      min_samples_leaf=int(p["min_samples_leaf"]),
                                      max_features=p["max_features"], class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=seed)
    if name == "svm_rbf":
        # Platt scaling (sigmoid) fitted with internal 3-fold CV = what SVC(probability=True) does,
        # written explicitly because that flag is deprecated in recent scikit-learn.
        svc = SVC(C=p["C"], gamma=p["gamma"], kernel="rbf", class_weight="balanced", random_state=seed)
        return Pipeline([("scale", StandardScaler()),
                         ("clf", CalibratedClassifierCV(svc, method="sigmoid", cv=3, ensemble=False))])
    raise KeyError(name)


# ------------------------------------------------------------------ data
def load_features(partition: str) -> pd.DataFrame:
    path = config.PROCESSED_DIR / f"features_{partition}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run scripts/05_build_features.py first")
    return pd.read_csv(path, dtype={"trip_id": str})


def valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["label_valid"] == 1]


def sample_observations(df: pd.DataFrame, frac: float, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Keep every candidate row of a random fraction of observations (per trip)."""
    if frac >= 1:
        return df
    obs = df[KEY].drop_duplicates()
    keep = obs.groupby("trip_id", group_keys=False).sample(frac=frac, random_state=seed)
    return df.merge(keep, on=KEY, how="inner")


def cap_rows(df: pd.DataFrame, max_rows: int | None, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Observation-level sample so that roughly max_rows candidate rows remain."""
    if max_rows is None or len(df) <= max_rows:
        return df
    return sample_observations(df, max_rows / len(df), seed)


def xy(df: pd.DataFrame):
    return df[FEATURES].to_numpy(np.float64), df["label"].to_numpy(np.int8)


def fit_model(name: str, train_df: pd.DataFrame, params: dict | None = None, seed: int = config.RANDOM_SEED):
    """Fit one model on valid rows, applying that model's row cap."""
    tr = cap_rows(train_df, SPECS[name].max_train_rows, seed)
    X, y = xy(tr)
    model = make_model(name, params)
    model.fit(X, y)
    return model, len(tr)


def predict_proba(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES].to_numpy(np.float64))[:, 1]
