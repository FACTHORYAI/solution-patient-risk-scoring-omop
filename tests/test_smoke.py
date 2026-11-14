"""One test per job on the synthetic sample, with references computed in plain Python."""

from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from patient_risk_scoring_omop import io
from patient_risk_scoring_omop.cli import main as cli_main
from patient_risk_scoring_omop.config import Settings
from patient_risk_scoring_omop.jobs import REGISTRY
from patient_risk_scoring_omop.jobs.build_features import Params as FeatureParams
from patient_risk_scoring_omop.jobs.build_outcomes import Params as OutcomeParams
from patient_risk_scoring_omop.jobs.build_target_cohort import Params as CohortParams
from patient_risk_scoring_omop.jobs.top_comorbidities import Params as ComorbidityParams
from patient_risk_scoring_omop.jobs.train_risk_model import FEATURES, RACE_FEATURE
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

RAW_TABLES = (
    "person",
    "condition_occurrence",
    "concept_ancestor",
    "visit_occurrence",
    "condition_era",
    "drug_exposure",
    "concept",
)
DERIVED_TABLES = (
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
SAMPLE_TABLES = RAW_TABLES + DERIVED_TABLES

HEART_FAILURE = 4229440
EMERGENCY_VISIT = 9203
WARFARIN = 40163554
ACETAMINOPHEN = 40221901


@pytest.fixture(scope="module")
def tables() -> dict[str, pa.Table]:
    return {name: io.read_table(name, SETTINGS) for name in SAMPLE_TABLES}


def run_job(name: str, tables: dict, params=None):
    spec = REGISTRY[name]
    inputs = {key: tables[key] for key in spec.inputs}
    return spec.run(inputs=inputs, params=params if params is not None else spec.params_cls())


def assert_tables_close(actual: pa.Table, expected: pa.Table) -> None:
    """Same schema and rows: strings, integers and dates equal, floats to a tolerance."""
    assert actual.schema.equals(expected.schema)
    assert actual.num_rows == expected.num_rows
    for field in actual.schema:
        got = actual.column(field.name).to_pylist()
        want = expected.column(field.name).to_pylist()
        if pa.types.is_floating(field.type):
            np.testing.assert_allclose(
                np.array(got, dtype=float), np.array(want, dtype=float), rtol=1e-6, atol=1e-9
            )
        else:
            assert got == want, field.name


# --- Plain-Python references, computed from the raw tables ---------------------------------


def days(count: int) -> timedelta:
    return timedelta(days=count)


def reference_cohort(
    tables: dict, min_observation_period: int = 1095, max_time_at_risk: int = 365
) -> dict[int, date]:
    """person id -> index date: first target diagnosis, enough observation before and after."""
    target = {
        row["descendant_concept_id"]
        for row in tables["concept_ancestor"].to_pylist()
        if row["ancestor_concept_id"] == HEART_FAILURE
    }
    first_diagnosis: dict[int, date] = {}
    for row in tables["condition_occurrence"].to_pylist():
        if row["condition_concept_id"] in target:
            person, start = row["person_id"], row["condition_start_date"]
            first_diagnosis[person] = min(start, first_diagnosis.get(person, start))
    first_visit: dict[int, date] = {}
    last_visit: dict[int, date] = {}
    for row in tables["visit_occurrence"].to_pylist():
        person, start = row["person_id"], row["visit_start_date"]
        first_visit[person] = min(start, first_visit.get(person, start))
        last_visit[person] = max(start, last_visit.get(person, start))
    return {
        person: index
        for person, index in first_diagnosis.items()
        if person in first_visit
        and index - first_visit[person] >= days(min_observation_period)
        and index + days(max_time_at_risk) < last_visit[person]
    }


def reference_outcomes(
    tables: dict, cohort: dict[int, date], min_time_at_risk: int = 7, max_time_at_risk: int = 365
) -> set[int]:
    """Cohort members with an emergency visit from index + min up to, not including, index + max."""
    outcome = set()
    for row in tables["visit_occurrence"].to_pylist():
        person = row["person_id"]
        if row["visit_concept_id"] != EMERGENCY_VISIT or person not in cohort:
            continue
        start = row["visit_start_date"]
        if (
            cohort[person] + days(min_time_at_risk)
            <= start
            < cohort[person] + days(max_time_at_risk)
        ):
            outcome.add(person)
    return outcome


def reference_features(tables: dict, cohort: dict[int, date], lookback_days: int = 1825) -> dict:
    """person id -> the features, measured before the index date."""
    people = {row["person_id"]: row for row in tables["person"].to_pylist()}
    features = {
        person: {
            "age_in_days": (index - date(people[person]["year_of_birth"], 1, 1)).days,
            "gender_concept_id": people[person]["gender_concept_id"],
            "race_concept_id": people[person]["race_concept_id"],
            "warfarin_quantity": 0.0,
            "acetaminophen_quantity": 0.0,
            "comorbidity_count": 0,
            "comorbidity_occurrences": 0,
            "prior_er_visits": 0,
        }
        for person, index in cohort.items()
    }
    for row in tables["drug_exposure"].to_pylist():
        person = row["person_id"]
        if person not in cohort:
            continue
        index, start = cohort[person], row["drug_exposure_start_date"]
        if index - days(lookback_days) <= start < index:
            if row["drug_concept_id"] == WARFARIN:
                features[person]["warfarin_quantity"] += row["quantity"]
            elif row["drug_concept_id"] == ACETAMINOPHEN:
                features[person]["acetaminophen_quantity"] += row["quantity"]
    conditions: dict[int, set[int]] = {}
    for row in tables["condition_era"].to_pylist():
        person = row["person_id"]
        if person not in cohort:
            continue
        index, start = cohort[person], row["condition_era_start_date"]
        if index - days(lookback_days) <= start < index:
            conditions.setdefault(person, set()).add(row["condition_concept_id"])
            features[person]["comorbidity_occurrences"] += row["condition_occurrence_count"]
    for person, concepts in conditions.items():
        features[person]["comorbidity_count"] = len(concepts)
    for row in tables["visit_occurrence"].to_pylist():
        person = row["person_id"]
        before_index = person in cohort and row["visit_start_date"] < cohort[person]
        if row["visit_concept_id"] == EMERGENCY_VISIT and before_index:
            features[person]["prior_er_visits"] += 1
    return features


# --- The sample ------------------------------------------------------------------------------


def test_sample_is_the_size_and_period_described(tables: dict) -> None:
    assert {name: tables[name].num_rows > 0 for name in SAMPLE_TABLES} == dict.fromkeys(
        SAMPLE_TABLES, True
    )
    assert tables["person"].num_rows == 900
    assert tables["target_cohort"].num_rows == 650
    assert 0.35 < tables["outcomes"].num_rows / 650 < 0.55
    dates = [
        row[column]
        for name, column in (
            ("visit_occurrence", "visit_start_date"),
            ("visit_occurrence", "visit_end_date"),
            ("condition_occurrence", "condition_start_date"),
            ("condition_era", "condition_era_start_date"),
            ("condition_era", "condition_era_end_date"),
            ("drug_exposure", "drug_exposure_start_date"),
            ("drug_exposure", "drug_exposure_end_date"),
        )
        for row in tables[name].to_pylist()
    ]
    assert date(2018, 1, 1) <= min(dates)
    assert max(dates) <= date(2025, 12, 31)


def test_sample_has_people_the_cohort_must_leave_out(tables: dict) -> None:
    cohort = reference_cohort(tables)
    diagnosed = reference_cohort(tables, min_observation_period=-100000, max_time_at_risk=-100000)
    too_early = reference_cohort(tables, max_time_at_risk=-100000)
    assert len(diagnosed) > len(too_early) > len(cohort) > 0
    assert len(diagnosed) < tables["person"].num_rows


def test_generator_reproduces_the_sample(tables: dict) -> None:
    path = REPO / "data" / "generate_sample.py"
    spec = importlib.util.spec_from_file_location("generate_sample", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generated = module.generate()
    assert set(generated) == set(SAMPLE_TABLES)
    for name in SAMPLE_TABLES:
        assert_tables_close(generated[name], tables[name])


# --- One test per job ------------------------------------------------------------------------


def test_build_target_cohort(tables: dict) -> None:
    cohort = run_job("build_target_cohort", tables)["target_cohort"]
    assert cohort.schema.equals(TARGET_COHORT)
    assert cohort.num_rows == 650
    assert_tables_close(cohort, tables["target_cohort"])
    expected = reference_cohort(tables)
    rows = cohort.to_pylist()
    assert {row["subject_id"]: row["cohort_end_date"] for row in rows} == expected
    assert all(row["cohort_start_date"] == row["cohort_end_date"] - days(1095) for row in rows)


def test_build_target_cohort_with_another_window(tables: dict) -> None:
    params = CohortParams(min_observation_period=365, max_time_at_risk=180)
    cohort = run_job("build_target_cohort", tables, params)["target_cohort"]
    expected = reference_cohort(tables, 365, 180)
    assert cohort.num_rows > 650
    assert {row["subject_id"]: row["cohort_end_date"] for row in cohort.to_pylist()} == expected


def test_build_target_cohort_rejects_bad_parameters(tables: dict) -> None:
    with pytest.raises(ValueError, match="min_observation_period"):
        run_job("build_target_cohort", tables, CohortParams(min_observation_period=-1))
    with pytest.raises(ValueError, match="max_time_at_risk"):
        run_job("build_target_cohort", tables, CohortParams(max_time_at_risk=0))


def test_build_outcomes(tables: dict) -> None:
    outcomes = run_job("build_outcomes", tables)["outcomes"]
    assert outcomes.schema.equals(OUTCOMES)
    assert_tables_close(outcomes, tables["outcomes"])
    expected = reference_outcomes(tables, reference_cohort(tables))
    assert 0 < len(expected) < 650
    assert {row["subject_id"] for row in outcomes.to_pylist()} == expected
    assert all(row["visited_emergency"] for row in outcomes.to_pylist())


def test_build_outcomes_with_another_window(tables: dict) -> None:
    params = OutcomeParams(min_time_at_risk=30, max_time_at_risk=180)
    outcomes = run_job("build_outcomes", tables, params)["outcomes"]
    expected = reference_outcomes(tables, reference_cohort(tables), 30, 180)
    assert 0 < len(expected) < tables["outcomes"].num_rows
    assert {row["subject_id"] for row in outcomes.to_pylist()} == expected


def test_build_outcomes_rejects_bad_window(tables: dict) -> None:
    with pytest.raises(ValueError, match="max_time_at_risk"):
        run_job("build_outcomes", tables, OutcomeParams(min_time_at_risk=30, max_time_at_risk=10))


@pytest.mark.parametrize("lookback_days", [1825, 365])
def test_build_features(tables: dict, lookback_days: int) -> None:
    features = run_job("build_features", tables, FeatureParams(lookback_days=lookback_days))
    features = features["person_features"]
    assert features.schema.equals(PERSON_FEATURES)
    assert features.num_rows == 650
    if lookback_days == 1825:
        assert_tables_close(features, tables["person_features"])
    expected = reference_features(tables, reference_cohort(tables), lookback_days)
    got = {row["subject_id"]: row for row in features.to_pylist()}
    assert set(got) == set(expected)
    for person, row in got.items():
        want = expected[person]
        for name, value in want.items():
            if isinstance(value, float):
                assert row[name] == pytest.approx(value), (person, name)
            else:
                assert row[name] == value, (person, name)
    assert any(row["comorbidity_count"] > 0 for row in got.values())
    assert any(row["warfarin_quantity"] > 0 for row in got.values())
    assert any(row["prior_er_visits"] > 0 for row in got.values())


def test_build_features_rejects_bad_lookback(tables: dict) -> None:
    with pytest.raises(ValueError, match="lookback_days"):
        run_job("build_features", tables, FeatureParams(lookback_days=0))


def test_build_training_data(tables: dict) -> None:
    training = run_job("build_training_data", tables)["training_data"]
    assert training.schema.equals(TRAINING_DATA)
    assert_tables_close(training, tables["training_data"])
    assert training.num_rows == 650
    outcome = reference_outcomes(tables, reference_cohort(tables))
    rows = training.to_pylist()
    assert {row["subject_id"] for row in rows if row["outcome"] == 1} == outcome
    assert set(training.column("outcome").to_pylist()) == {0, 1}
    assert RACE_FEATURE in training.column_names


def test_train_risk_model(tables: dict) -> None:
    outputs = run_job("train_risk_model", tables)

    metrics = outputs["model_metrics"]
    assert metrics.schema.equals(MODEL_METRICS)
    assert_tables_close(metrics, tables["model_metrics"])
    rows = {row["split"]: row for row in metrics.to_pylist()}
    assert set(rows) == {"train", "validation", "test"}
    assert sum(row["notes"] for row in rows.values()) == 650
    assert rows["test"]["notes"] == 130
    assert all(row["positives"] > 0 for row in rows.values())
    assert rows["validation"]["roc_auc"] > 0.65
    assert rows["test"]["roc_auc"] > 0.65

    importance = outputs["feature_importance"]
    assert importance.schema.equals(FEATURE_IMPORTANCE)
    assert_tables_close(importance, tables["feature_importance"])
    names = [row["feature"] for row in importance.to_pylist()]
    assert RACE_FEATURE not in names
    assert set(names) == set(FEATURES) - {RACE_FEATURE}
    assert [row["rank"] for row in importance.to_pylist()] == list(range(1, len(names) + 1))
    coefficients = [abs(row["coefficient"]) for row in importance.to_pylist()]
    assert coefficients == sorted(coefficients, reverse=True)

    predictions = outputs["model_predictions"]
    assert predictions.schema.equals(MODEL_PREDICTIONS)
    assert_tables_close(predictions, tables["model_predictions"])
    assert predictions.num_rows == 650
    subjects = predictions.column("subject_id").to_pylist()
    assert sorted(subjects) == sorted(tables["training_data"].column("subject_id").to_pylist())
    assert all(0.0 <= row["predicted_probability"] <= 1.0 for row in predictions.to_pylist())


def test_train_risk_model_without_race_by_default_and_with_it_on_request(tables: dict) -> None:
    assert ModelParams().include_race is False
    default = run_job("train_risk_model", tables)["feature_importance"]
    with_race = run_job("train_risk_model", tables, ModelParams(include_race=True))
    with_race = with_race["feature_importance"]
    assert RACE_FEATURE not in default.column("feature").to_pylist()
    assert RACE_FEATURE in with_race.column("feature").to_pylist()
    assert with_race.num_rows == default.num_rows + 1
    # Gender stays a feature either way.
    assert "gender_concept_id" in default.column("feature").to_pylist()


def test_train_risk_model_is_reproducible_with_the_same_seed(tables: dict) -> None:
    first = run_job("train_risk_model", tables, ModelParams(random_state=7))
    second = run_job("train_risk_model", tables, ModelParams(random_state=7))
    other = run_job("train_risk_model", tables, ModelParams(random_state=8))
    for name in first:
        assert_tables_close(first[name], second[name])

    def split_of(outputs: dict) -> dict[int, str]:
        rows = outputs["model_predictions"].to_pylist()
        return {row["subject_id"]: row["split"] for row in rows}

    assert split_of(first) != split_of(other)


def test_train_risk_model_rejects_bad_split(tables: dict) -> None:
    with pytest.raises(ValueError, match="leave room"):
        run_job("train_risk_model", tables, ModelParams(test_fraction=0.5, validation_fraction=0.5))


def test_risk_summary(tables: dict) -> None:
    bands = run_job("risk_summary", tables)["risk_bands"]
    assert bands.schema.equals(RISK_BANDS)
    assert_tables_close(bands, tables["risk_bands"])
    rows = bands.to_pylist()
    assert len(rows) >= 4
    assert sum(row["notes"] for row in rows) == 650
    probability = [row["mean_probability"] for row in rows]
    observed = [row["observed_rate"] for row in rows]
    assert probability == sorted(probability)
    assert len(set(probability)) == len(probability)
    # The observed rate rises from the lowest band to the highest.
    assert observed == sorted(observed)
    assert observed[-1] - observed[0] > 0.4
    for row in rows:
        low = float(row["risk_band"].split("-")[0])
        assert low <= row["mean_probability"] <= low + 0.2 + 1e-9


def test_top_comorbidities(tables: dict) -> None:
    comorbidities = run_job("top_comorbidities", tables)["top_comorbidities"]
    assert comorbidities.schema.equals(TOP_COMORBIDITIES)
    assert_tables_close(comorbidities, tables["top_comorbidities"])
    rows = comorbidities.to_pylist()
    assert len(rows) == 5
    patients = [row["patients"] for row in rows]
    assert patients == sorted(patients, reverse=True)
    assert all(row["condition_name"] for row in rows)
    # The most frequent condition, counted in plain Python.
    cohort = reference_cohort(tables)
    seen: dict[int, set[int]] = {}
    for row in tables["condition_era"].to_pylist():
        person = row["person_id"]
        if (
            person in cohort
            and cohort[person] - days(1825) <= row["condition_era_start_date"] < cohort[person]
        ):
            seen.setdefault(row["condition_concept_id"], set()).add(person)
    assert rows[0]["patients"] == max(len(people) for people in seen.values())
    with pytest.raises(ValueError, match="limit"):
        run_job("top_comorbidities", tables, ComorbidityParams(limit=0))


# --- The command line ------------------------------------------------------------------------


def test_every_job_runs_through_the_command_line(
    tables: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    (tmp_path / "patient_risk_scoring_omop.toml").write_text(
        f'[settings]\nsample_dir = "{(REPO / "data" / "sample").as_posix()}"\n'
        f'output_dir = "{output.as_posix()}"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    for job in REGISTRY:
        assert cli_main([job]) == 0
    for name in DERIVED_TABLES:
        assert_tables_close(pq.read_table(output / f"{name}.parquet"), tables[name])
    assert cli_main(["train_risk_model", "--include-race"]) == 0
    importance = pq.read_table(output / "feature_importance.parquet")
    assert RACE_FEATURE in importance.column("feature").to_pylist()
