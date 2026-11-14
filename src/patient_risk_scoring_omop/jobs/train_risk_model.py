"""Train and evaluate a logistic-regression risk model on the training data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from patient_risk_scoring_omop.jobs._duck import align
from patient_risk_scoring_omop.schemas import FEATURE_IMPORTANCE, MODEL_METRICS, MODEL_PREDICTIONS

INPUTS = ("training_data",)
OUTPUTS = ("model_metrics", "feature_importance", "model_predictions")

# Every column of training_data that can be a model feature.
FEATURES = (
    "age_in_days",
    "gender_concept_id",
    "race_concept_id",
    "warfarin_quantity",
    "acetaminophen_quantity",
    "comorbidity_count",
    "comorbidity_occurrences",
    "prior_er_visits",
)
RACE_FEATURE = "race_concept_id"
MINIMUM_ROWS = 30


def model_features(include_race: bool) -> tuple[str, ...]:
    """The features the model is trained on; race is left out unless asked for."""
    return FEATURES if include_race else tuple(name for name in FEATURES if name != RACE_FEATURE)


@dataclass(frozen=True)
class Params:
    """Parameters for the risk model."""

    test_fraction: float = 0.2
    validation_fraction: float = 0.2
    random_state: int = 42
    max_iter: int = 1000
    threshold: float = 0.5
    include_race: bool = False


def run(inputs: dict[str, pa.Table], params: Params) -> dict[str, pa.Table]:
    """Return per-split metrics, the model coefficients and the per-person predictions.

    Race (`race_concept_id`) stays in `training_data` so a fairness check is possible, but it is
    not a model feature unless `include_race` is true.
    """
    rows = inputs["training_data"].to_pylist()
    _check_params(params)
    names = model_features(params.include_race)
    if len(rows) < MINIMUM_ROWS:
        message = (
            f"training_data has {len(rows)} rows; train_risk_model needs at least {MINIMUM_ROWS}"
        )
        raise ValueError(message)
    features = np.array([[float(row[name]) for name in names] for row in rows])
    outcomes = np.array([int(row["outcome"]) for row in rows])
    subject_ids = np.array([row["subject_id"] for row in rows])
    if len(set(outcomes.tolist())) < 2:
        message = "training_data has only one outcome class; the model cannot be trained"
        raise ValueError(message)

    positions = np.arange(len(rows))
    train_val, test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
        features,
        outcomes,
        positions,
        test_size=params.test_fraction,
        random_state=params.random_state,
        stratify=outcomes,
    )
    validation_size = params.validation_fraction / (1.0 - params.test_fraction)
    train, validation, y_train, y_validation, idx_train, idx_validation = train_test_split(
        train_val,
        y_train_val,
        idx_train_val,
        test_size=validation_size,
        random_state=params.random_state,
        stratify=y_train_val,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=params.max_iter,
                    random_state=params.random_state,
                ),
            ),
        ]
    )
    model.fit(train, y_train)

    splits = {
        "train": (train, y_train, idx_train),
        "validation": (validation, y_validation, idx_validation),
        "test": (test, y_test, idx_test),
    }
    metrics_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for split_name, (split_features, split_outcomes, split_positions) in splits.items():
        probabilities = model.predict_proba(split_features)[:, 1]
        predicted = (probabilities >= params.threshold).astype(int)
        metrics_rows.append(
            {
                "split": split_name,
                "notes": len(split_outcomes),
                "positives": int(split_outcomes.sum()),
                "accuracy": float(accuracy_score(split_outcomes, predicted)),
                "precision": float(precision_score(split_outcomes, predicted, zero_division=0)),
                "recall": float(recall_score(split_outcomes, predicted, zero_division=0)),
                "f1": float(f1_score(split_outcomes, predicted, zero_division=0)),
                "roc_auc": (
                    float(roc_auc_score(split_outcomes, probabilities))
                    if len(set(split_outcomes.tolist())) == 2
                    else None
                ),
            }
        )
        for position, outcome, probability, label in zip(
            split_positions, split_outcomes, probabilities, predicted, strict=True
        ):
            prediction_rows.append(
                {
                    "subject_id": int(subject_ids[position]),
                    "split": split_name,
                    "outcome": int(outcome),
                    "predicted_probability": float(probability),
                    "predicted_outcome": int(label),
                }
            )

    coefficients = model.named_steps["classifier"].coef_[0]
    importance_rows = [
        {
            "feature": name,
            "coefficient": float(coefficient),
            "odds_ratio": float(np.exp(coefficient)),
        }
        for name, coefficient in zip(names, coefficients, strict=True)
    ]
    importance_rows.sort(key=lambda row: abs(row["coefficient"]), reverse=True)
    for rank, row in enumerate(importance_rows, start=1):
        row["rank"] = rank

    return {
        "model_metrics": align(
            pa.Table.from_pylist(metrics_rows, schema=MODEL_METRICS), MODEL_METRICS
        ),
        "feature_importance": align(
            pa.Table.from_pylist(importance_rows, schema=FEATURE_IMPORTANCE), FEATURE_IMPORTANCE
        ),
        "model_predictions": align(
            pa.Table.from_pylist(prediction_rows, schema=MODEL_PREDICTIONS), MODEL_PREDICTIONS
        ),
    }


def _check_params(params: Params) -> None:
    """Raise when a split parameter cannot produce three non-empty sets."""
    if not 0 < params.test_fraction < 1:
        message = f"Parameter test_fraction is {params.test_fraction}; it must be between 0 and 1"
        raise ValueError(message)
    if not 0 < params.validation_fraction < 1:
        message = (
            f"Parameter validation_fraction is {params.validation_fraction}; "
            "it must be between 0 and 1"
        )
        raise ValueError(message)
    if params.test_fraction + params.validation_fraction >= 1:
        message = (
            f"test_fraction ({params.test_fraction}) + validation_fraction "
            f"({params.validation_fraction}) must leave room for a training set"
        )
        raise ValueError(message)
    if not 0 <= params.threshold <= 1:
        message = f"Parameter threshold is {params.threshold}; it must be between 0 and 1"
        raise ValueError(message)
    if params.max_iter < 1:
        message = f"Parameter max_iter is {params.max_iter}; it must be at least 1"
        raise ValueError(message)
