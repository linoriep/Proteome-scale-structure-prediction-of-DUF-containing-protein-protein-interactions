from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "combined_score",
    "neighborhood",
    "neighborhood_transferred",
    "fusion",
    "cooccurence",
    "homology",
    "coexpression",
    "coexpression_transferred",
    "experiments",
    "experiments_transferred",
    "database",
    "database_transferred",
    "textmining",
    "textmining_transferred",
]
MODULE_FEATURES = ["max_combined", *[f"max_{name}" for name in FEATURES[1:]]]
LABEL = "liberal_high_confidence"
RANDOM_STATE = 13


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    C=0.003,
                    class_weight={0: 1, 1: 12},
                    solver="lbfgs",
                    max_iter=4000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def load_training_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    data = pd.read_csv(path, sep="\t", low_memory=False)
    missing = [column for column in ["prediction_name", LABEL, *FEATURES] if column not in data]
    if missing:
        raise ValueError(f"Missing training columns: {missing}")
    features = data[FEATURES].apply(pd.to_numeric, errors="coerce") / 1000.0
    labels = data[LABEL].astype(str).str.lower().map({"true": 1, "false": 0})
    labels = labels.fillna(pd.to_numeric(data[LABEL], errors="coerce")).astype(int)
    return features, labels, data["prediction_name"].astype(str)


def cross_validate(
    features: pd.DataFrame, labels: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    scores = np.full(len(features), np.nan)
    folds = np.full(len(features), -1, dtype=int)
    metrics = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_index, test_index) in enumerate(splitter.split(features, labels), start=1):
        pipeline = make_pipeline()
        pipeline.fit(features.iloc[train_index], labels.iloc[train_index])
        fold_scores = pipeline.predict_proba(features.iloc[test_index])[:, 1]
        scores[test_index] = fold_scores
        folds[test_index] = fold
        metrics.append(
            {
                "fold": fold,
                "test_rows": len(test_index),
                "test_positives": int(labels.iloc[test_index].sum()),
                "average_precision": average_precision_score(labels.iloc[test_index], fold_scores),
                "auroc": roc_auc_score(labels.iloc[test_index], fold_scores),
            }
        )
    pooled = {
        "average_precision": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
    }
    return pd.DataFrame({"fold": folds, "score": scores}), pd.DataFrame(metrics), pooled


def fit_model(features: pd.DataFrame, labels: pd.Series) -> Pipeline:
    pipeline = make_pipeline()
    pipeline.fit(features, labels)
    return pipeline


def coefficient_table(pipeline: Pipeline) -> pd.DataFrame:
    model = pipeline.named_steps["logistic_regression"]
    return pd.DataFrame({"feature": FEATURES, "standardized_coefficient": model.coef_[0]})


def score_modules(pipeline: Pipeline, modules: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in MODULE_FEATURES if column not in modules]
    if missing:
        raise ValueError(f"Missing module feature columns: {missing}")
    matrix = modules[MODULE_FEATURES].apply(pd.to_numeric, errors="coerce") / 1000.0
    matrix.columns = FEATURES
    scored = modules.copy()
    scored["l2_model_score"] = pipeline.predict_proba(matrix)[:, 1]
    return scored


def save_model(pipeline: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
