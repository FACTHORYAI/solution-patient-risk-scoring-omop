"""One smoke test per job, on the synthetic sample data."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from patient_risk_scoring_omop import io
from patient_risk_scoring_omop.config import Settings
from patient_risk_scoring_omop.jobs import REGISTRY
from patient_risk_scoring_omop.jobs.build_outcomes import Params as OutcomeParams
from patient_risk_scoring_omop.jobs.train_risk_model import FEATURES
from patient_risk_scoring_omop.jobs.train_risk_model import Params as ModelParams
from patient_risk_scoring_omop.schemas import (
    FEATURE_IMPORTANCE,
    MODEL_METRICS,
    MODEL_PREDICTIONS,
    OUTCOMES,
    PERSON_FEATURES,
    RISK_BANDS,
    TARGET_COHORT,
    TOP_COMORBIDITIES,
    TRAINING_DATA,
)

REPO = Path(__file__).resolve().parent.parent
SETTINGS = Settings(sample_dir=REPO / "data" / "sample", output_dir=REPO / "output")

SAMPLE_TABLES = (
    "person",
    "condition_occurrence",
    "concept_ancestor",
    "visit_occurrence",
    "condition_era",
    "drug_exposure",
    "concept",
    "target_cohort",
    "outcomes",
    "person_features",
    "training_data",
    "model_metrics",
    "feature_importance",
    "model_predictions",
    "risk_bands",
    "top_comorbidities",
)


@pytest.fixture(scope="module")
def tables() -> dict[str, object]:
    return {name: io.read_table(name, SETTINGS) for name in SAMPLE_TABLES}


def run_job(name: str, inputs: dict, params=None):
    spec = REGISTRY[name]
    return spec.run(inputs=inputs, params=params if params is not None else spec.params_cls())


def test_build_target_cohort(tables: dict) -> None:
    outputs = run_job(
        "build_target_cohort",
        {
            "person": tables["person"],
            "condition_occurrence": tables["condition_occurrence"],
            "concept_ancestor": tables["concept_ancestor"],
            "visit_occurrence": tables["visit_occurrence"],
        },
    )
    cohort = outputs["target_cohort"]
    assert cohort.schema.equals(TARGET_COHORT)
    assert cohort.equals(tables["target_cohort"].select(TARGET_COHORT.names))
    assert cohort.num_rows == 171
    first = cohort.to_pylist()[0]
    assert first["cohort_start_date"] == first["cohort_end_date"] - timedelta(days=1095)


def test_build_outcomes(tables: dict) -> None:
    outputs = run_job(
        "build_outcomes",
        {"target_cohort": tables["target_cohort"], "visit_occurrence": tables["visit_occurrence"]},
    )
    outcomes = outputs["outcomes"]
    assert outcomes.schema.equals(OUTCOMES)
    assert outcomes.equals(tables["outcomes"].select(OUTCOMES.names))
    assert outcomes.num_rows == 70
    assert all(row["visited_emergency"] for row in outcomes.to_pylist())


def test_build_outcomes_rejects_bad_window(tables: dict) -> None:
    with pytest.raises(ValueError, match="max_time_at_risk"):
        run_job(
            "build_outcomes",
            {
                "target_cohort": tables["target_cohort"],
                "visit_occurrence": tables["visit_occurrence"],
            },
            OutcomeParams(min_time_at_risk=30, max_time_at_risk=10),
        )


def test_build_features(tables: dict) -> None:
    outputs = run_job(
        "build_features",
        {
            "target_cohort": tables["target_cohort"],
            "person": tables["person"],
            "condition_era": tables["condition_era"],
            "drug_exposure": tables["drug_exposure"],
            "visit_occurrence": tables["visit_occurrence"],
        },
    )
    features = outputs["person_features"]
    assert features.schema.equals(PERSON_FEATURES)
    assert features.equals(tables["person_features"].select(PERSON_FEATURES.names))
    assert features.num_rows == 171


def test_build_training_data(tables: dict) -> None:
    outputs = run_job(
        "build_training_data",
        {
            "target_cohort": tables["target_cohort"],
            "outcomes": tables["outcomes"],
            "person_features": tables["person_features"],
        },
    )
    training = outputs["training_data"]
    assert training.schema.equals(TRAINING_DATA)
    assert training.equals(tables["training_data"].select(TRAINING_DATA.names))
    assert training.num_rows == 171
    assert set(training.column("outcome").to_pylist()) == {0, 1}


def test_train_risk_model(tables: dict) -> None:
    outputs = run_job("train_risk_model", {"training_data": tables["training_data"]})

    metrics = outputs["model_metrics"]
    assert metrics.schema.equals(MODEL_METRICS)
    assert metrics.equals(tables["model_metrics"].select(MODEL_METRICS.names))
    rows = {row["split"]: row for row in metrics.to_pylist()}
    assert set(rows) == {"train", "validation", "test"}
    assert sum(row["notes"] for row in rows.values()) == 171
    assert rows["train"]["roc_auc"] > 0.6

    importance = outputs["feature_importance"]
    assert importance.schema.equals(FEATURE_IMPORTANCE)
    assert importance.equals(tables["feature_importance"].select(FEATURE_IMPORTANCE.names))
    assert importance.num_rows == len(FEATURES)
    assert [row["rank"] for row in importance.to_pylist()] == list(range(1, len(FEATURES) + 1))
    assert {row["feature"] for row in importance.to_pylist()} == set(FEATURES)

    predictions = outputs["model_predictions"]
    assert predictions.schema.equals(MODEL_PREDICTIONS)
    assert predictions.equals(tables["model_predictions"].select(MODEL_PREDICTIONS.names))
    assert predictions.num_rows == 171
    assert all(0.0 <= row["predicted_probability"] <= 1.0 for row in predictions.to_pylist())


def test_train_risk_model_rejects_bad_split(tables: dict) -> None:
    with pytest.raises(ValueError, match="leave room"):
        run_job(
            "train_risk_model",
            {"training_data": tables["training_data"]},
            ModelParams(test_fraction=0.5, validation_fraction=0.5),
        )


def test_risk_summary(tables: dict) -> None:
    outputs = run_job("risk_summary", {"model_predictions": tables["model_predictions"]})
    bands = outputs["risk_bands"]
    assert bands.schema.equals(RISK_BANDS)
    assert bands.equals(tables["risk_bands"].select(RISK_BANDS.names))
    rows = bands.to_pylist()
    assert 1 <= len(rows) <= 5
    assert sum(row["notes"] for row in rows) == 171
    assert all(isinstance(row["observed_rate"], float) for row in rows)


def test_top_comorbidities(tables: dict) -> None:
    outputs = run_job(
        "top_comorbidities",
        {
            "condition_era": tables["condition_era"],
            "target_cohort": tables["target_cohort"],
            "concept": tables["concept"],
        },
    )
    comorbidities = outputs["top_comorbidities"]
    assert comorbidities.schema.equals(TOP_COMORBIDITIES)
    assert comorbidities.equals(tables["top_comorbidities"].select(TOP_COMORBIDITIES.names))
    assert comorbidities.num_rows == 5
    patients = [row["patients"] for row in comorbidities.to_pylist()]
    assert patients == sorted(patients, reverse=True)
    assert all(row["condition_name"] for row in comorbidities.to_pylist())


def test_cohort_dates_are_dates(tables: dict) -> None:
    first = tables["target_cohort"].to_pylist()[0]
    assert isinstance(first["cohort_start_date"], date)
    assert first["cohort_start_date"] < first["cohort_end_date"]
