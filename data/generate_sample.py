"""Generate the synthetic sample: an invented OMOP 5.3 extract of heart-failure patients.

Everything here is invented. No real patient, record or population is used. Run it from the
repository root:

    uv run python data/generate_sample.py

It writes data/sample/*.parquet: the raw tables the pipeline reads (person,
condition_occurrence, concept_ancestor, visit_occurrence, condition_era, drug_exposure,
concept) and then, by running the repository's own jobs on them with their default
parameters, every derived table (target_cohort, outcomes, person_features, training_data,
model_metrics, feature_importance, model_predictions, risk_bands, top_comorbidities). The
same seed always gives the same files.

How the raw tables are made:

- 900 persons in four groups. 650 are diagnosed with heart failure (the target condition, or
  its descendant) with at least three years of observation before the diagnosis and a visit
  more than a year after it: they form the target cohort. 90 are diagnosed too early in their
  observation, 70 have no visit more than a year after the diagnosis, and 90 never have a
  heart-failure diagnosis; the jobs must leave these out. Dates run from 2018 to 2025.
- Each person has outpatient visits, condition eras for up to six comorbidities, warfarin and
  acetaminophen exposures and emergency-room visits, some of them after the diagnosis or on
  its boundaries, so that the time windows of the jobs matter.
- The outcome, an emergency-room visit from 7 days to just under a year after the diagnosis,
  is drawn for each cohort member with a probability that is a logistic function of the age at
  diagnosis, the number of comorbidities, their occurrences, the emergency-room visits before
  the diagnosis and warfarin exposure, plus noise. Gender and race have no effect on it.
  The signal is built in so that the model has something to learn: how well it performs on
  this sample says nothing about any real population or model.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa

from patient_risk_scoring_omop import io
from patient_risk_scoring_omop.config import Settings
from patient_risk_scoring_omop.jobs import REGISTRY
from patient_risk_scoring_omop.schemas import (
    CONCEPT,
    CONCEPT_ANCESTOR,
    CONDITION_ERA,
    CONDITION_OCCURRENCE,
    DRUG_EXPOSURE,
    PERSON,
    VISIT_OCCURRENCE,
)

SEED = 20251120
REPO = Path(__file__).resolve().parent.parent

DATA_START = date(2018, 1, 1)
DATA_END = date(2025, 12, 31)

# Persons per group (900 in all).
COHORT, TOO_EARLY, NO_FOLLOW_UP, NO_DIAGNOSIS = 650, 90, 70, 90

HEART_FAILURE = 4229440
HEART_FAILURE_UNSPECIFIED = 319835  # a descendant of HEART_FAILURE
EMERGENCY_VISIT = 9203
OUTPATIENT_VISIT = 9202
WARFARIN = 40163554
ACETAMINOPHEN = 40221901
COMORBIDITIES = (201826, 316866, 255573, 320128, 201820, 192279)
MALE, FEMALE = 8507, 8532
RACES = (8527, 8516, 8515)

CONCEPTS = [
    (HEART_FAILURE, "Congestive heart failure"),
    (HEART_FAILURE_UNSPECIFIED, "Congestive heart failure, unspecified"),
    (EMERGENCY_VISIT, "Emergency room visit"),
    (OUTPATIENT_VISIT, "Outpatient visit"),
    (WARFARIN, "Warfarin"),
    (ACETAMINOPHEN, "Acetaminophen"),
    (201826, "Type 2 diabetes mellitus"),
    (316866, "Hypertensive disorder"),
    (255573, "Chronic obstructive lung disease"),
    (320128, "Essential hypertension"),
    (201820, "Diabetes mellitus"),
    (192279, "Chronic kidney disease"),
]

# The built-in signal: the outcome's log-odds. Gender, race and acetaminophen have no effect.
INTERCEPT = -2.2
PER_DECADE_OF_AGE = 0.55  # per ten years of age at diagnosis, around 68
PER_COMORBIDITY = 0.35  # per distinct comorbidity in the five years before the diagnosis
PER_OCCURRENCE = 0.10  # per recorded occurrence of those comorbidities
PER_PRIOR_ER_VISIT = 0.50  # per emergency-room visit before the diagnosis
WARFARIN_EXPOSED = 0.90  # warfarin in the five years before the diagnosis
NOISE_SD = 0.5


def _poisson(rng: random.Random, mean: float) -> int:
    limit, count, product = math.exp(-mean), 0, rng.random()
    while product > limit:
        count += 1
        product *= rng.random()
    return count


def _day(rng: random.Random, first: date, last: date) -> date:
    """A day between two dates, both included."""
    return first + timedelta(days=rng.randint(0, (last - first).days))


def _make_person(rng: random.Random, person_id: int, group: str) -> dict:
    """One person's raw rows, and the facts the outcome is drawn from."""
    if group == "no_follow_up":
        index = _day(rng, date(2021, 2, 1), date(2025, 6, 30))
    elif group == "too_early":
        index = _day(rng, date(2019, 6, 1), date(2024, 11, 30))
    else:
        index = _day(rng, date(2021, 2, 1), date(2024, 11, 30))
    age = min(92.0, max(35.0, rng.gauss(68, 11)))
    person = {
        "person_id": person_id,
        "gender_concept_id": MALE if rng.random() < 0.55 else FEMALE,
        "year_of_birth": index.year - round(age),
        "race_concept_id": rng.choices(RACES, weights=[70, 20, 10])[0],
    }

    # Observation: outpatient visits from a first to a last one.
    if group == "too_early":
        first_visit = max(
            DATA_START + timedelta(days=1), index - timedelta(days=rng.randint(60, 1000))
        )
    else:
        latest_first = max(
            index - timedelta(days=1096 + rng.randint(0, 200)), DATA_START + timedelta(days=1)
        )
        first_visit = _day(rng, DATA_START + timedelta(days=1), latest_first)
    if group == "no_follow_up":
        last_visit = min(index + timedelta(days=rng.randint(30, 330)), DATA_END)
    else:
        last_visit = min(index + timedelta(days=366 + rng.randint(0, 120)), DATA_END)
    visit_days = {first_visit, last_visit}
    for _ in range(rng.randint(7, 17)):
        visit_days.add(_day(rng, first_visit, last_visit))
    visits = [(OUTPATIENT_VISIT, day, day) for day in visit_days]

    # Comorbidities: older patients have more of them; one or two eras each, in the five years
    # before the diagnosis, and sometimes one era after it that no feature may count.
    history_start = max(index - timedelta(days=1825), DATA_START)
    chance = min(0.7, max(0.08, 0.28 + 0.008 * (age - 68)))
    chosen = [concept for concept in COMORBIDITIES if rng.random() < chance]
    eras = []
    for concept in chosen:
        for _ in range(1 if rng.random() < 0.85 else 2):
            start = _day(rng, history_start, index - timedelta(days=30))
            eras.append((concept, start, 1 + min(7, int(rng.expovariate(1 / 1.8)))))
    if rng.random() < 0.25:
        after = index + timedelta(days=rng.randint(30, 300))
        eras.append((rng.choice(COMORBIDITIES), min(after, DATA_END), rng.randint(1, 4)))
    in_history = [era for era in eras if era[1] < index]
    comorbidity_count = len({era[0] for era in in_history})
    occurrences = sum(era[2] for era in in_history)

    # Drugs: warfarin and acetaminophen fills in the five years before the diagnosis, and a few
    # on the diagnosis day or after it, which no feature may count.
    drugs = []
    warfarin = rng.random() < 0.30
    if warfarin:
        for _ in range(rng.randint(1, 4)):
            start = index - timedelta(days=rng.randint(1, 1500))
            drugs.append(
                (WARFARIN, max(start, DATA_START), float(rng.choice([14, 28, 30, 60, 90])))
            )
    elif rng.random() < 0.15:
        drugs.append((WARFARIN, min(index + timedelta(days=rng.randint(1, 200)), DATA_END), 30.0))
    if rng.random() < 0.05:
        drugs.append((WARFARIN, index, 28.0))
    if rng.random() < 0.45:
        for _ in range(rng.randint(1, 3)):
            start = index - timedelta(days=rng.randint(1, 1500))
            drugs.append(
                (ACETAMINOPHEN, max(start, DATA_START), float(rng.choice([10, 20, 50, 100])))
            )

    # Emergency-room visits before the diagnosis.
    prior = _poisson(rng, 0.35 + 0.2 * comorbidity_count)
    emergencies = [
        _day(rng, first_visit + timedelta(days=1), index - timedelta(days=1)) for _ in range(prior)
    ]

    # The outcome, from the built-in signal.
    log_odds = (
        INTERCEPT
        + PER_DECADE_OF_AGE * (age - 68) / 10
        + PER_COMORBIDITY * comorbidity_count
        + PER_OCCURRENCE * occurrences
        + PER_PRIOR_ER_VISIT * prior
        + (WARFARIN_EXPOSED if warfarin else 0.0)
        + rng.gauss(0, NOISE_SD)
    )
    outcome = rng.random() < 1 / (1 + math.exp(-log_odds))
    if outcome:
        emergencies.append(index + timedelta(days=rng.randint(7, 364)))
        if rng.random() < 0.3:
            emergencies.append(index + timedelta(days=rng.randint(7, 364)))
    else:
        # Visits next to the risk window that must not count: before it starts (day 0 to 6) and
        # after it ends (day 365 on).
        if rng.random() < 0.08:
            emergencies.append(index + timedelta(days=rng.randint(0, 6)))
        if group != "no_follow_up" and rng.random() < 0.08:
            emergencies.append(min(index + timedelta(days=rng.randint(365, 500)), DATA_END))
    if group == "no_follow_up":
        # No visit may reach a year after the diagnosis, nor the end of the data.
        latest = min(index + timedelta(days=330), DATA_END)
        emergencies = [day for day in emergencies if day <= latest]
    visits += [
        (EMERGENCY_VISIT, day, min(day + timedelta(days=rng.randint(0, 3)), DATA_END))
        for day in emergencies
    ]

    # Heart failure: the first record, sometimes under the descendant concept, and repeats.
    conditions = [(concept, start) for concept, start, _ in eras]
    if group != "no_diagnosis":
        first = HEART_FAILURE if rng.random() < 0.7 else HEART_FAILURE_UNSPECIFIED
        conditions.append((first, index))
        if rng.random() < 0.6:
            conditions.append(
                (HEART_FAILURE, min(index + timedelta(days=rng.randint(20, 400)), DATA_END))
            )
    return {
        "person": person,
        "visits": visits,
        "eras": eras,
        "conditions": conditions,
        "drugs": drugs,
    }


def _raw_tables(rng: random.Random) -> dict[str, pa.Table]:
    groups = (
        ["cohort"] * COHORT
        + ["too_early"] * TOO_EARLY
        + ["no_follow_up"] * NO_FOLLOW_UP
        + ["no_diagnosis"] * NO_DIAGNOSIS
    )
    rng.shuffle(groups)
    persons, visits, eras, conditions, drugs = [], [], [], [], []
    for position, group in enumerate(groups):
        person_id = 100001 + position
        made = _make_person(rng, person_id, group)
        persons.append(made["person"])
        for concept, start, end in sorted(made["visits"], key=lambda v: (v[1], v[0])):
            visits.append(
                {
                    "visit_occurrence_id": len(visits) + 1,
                    "person_id": person_id,
                    "visit_concept_id": concept,
                    "visit_start_date": start,
                    "visit_end_date": end,
                }
            )
        for concept, start, count in sorted(made["eras"], key=lambda e: (e[1], e[0])):
            end = min(start + timedelta(days=rng.randint(30, 400)), DATA_END)
            eras.append(
                {
                    "condition_era_id": len(eras) + 1,
                    "person_id": person_id,
                    "condition_concept_id": concept,
                    "condition_era_start_date": start,
                    "condition_era_end_date": end,
                    "condition_occurrence_count": count,
                }
            )
        for concept, start in sorted(made["conditions"], key=lambda c: (c[1], c[0])):
            conditions.append(
                {
                    "condition_occurrence_id": len(conditions) + 1,
                    "person_id": person_id,
                    "condition_concept_id": concept,
                    "condition_start_date": start,
                }
            )
        for concept, start, quantity in sorted(made["drugs"], key=lambda d: (d[1], d[0])):
            drugs.append(
                {
                    "drug_exposure_id": len(drugs) + 1,
                    "person_id": person_id,
                    "drug_concept_id": concept,
                    "drug_exposure_start_date": start,
                    "drug_exposure_end_date": min(start + timedelta(days=int(quantity)), DATA_END),
                    "quantity": quantity,
                }
            )
    return {
        "person": pa.Table.from_pylist(persons, schema=PERSON),
        "condition_occurrence": pa.Table.from_pylist(conditions, schema=CONDITION_OCCURRENCE),
        "concept_ancestor": pa.Table.from_pylist(
            [
                {"ancestor_concept_id": HEART_FAILURE, "descendant_concept_id": HEART_FAILURE},
                {
                    "ancestor_concept_id": HEART_FAILURE,
                    "descendant_concept_id": HEART_FAILURE_UNSPECIFIED,
                },
            ],
            schema=CONCEPT_ANCESTOR,
        ),
        "visit_occurrence": pa.Table.from_pylist(visits, schema=VISIT_OCCURRENCE),
        "condition_era": pa.Table.from_pylist(eras, schema=CONDITION_ERA),
        "drug_exposure": pa.Table.from_pylist(drugs, schema=DRUG_EXPOSURE),
        "concept": pa.Table.from_pylist(
            [{"concept_id": cid, "concept_name": name} for cid, name in CONCEPTS], schema=CONCEPT
        ),
    }


def _run(name: str, tables: dict[str, pa.Table]) -> dict[str, pa.Table]:
    spec = REGISTRY[name]
    return spec.run(inputs={key: tables[key] for key in spec.inputs}, params=spec.params_cls())


def generate() -> dict[str, pa.Table]:
    """Every table of the sample: the raw ones, then what the repository's own jobs derive."""
    tables = _raw_tables(random.Random(SEED))
    for name in (
        "build_target_cohort",
        "build_outcomes",
        "build_features",
        "build_training_data",
        "train_risk_model",
        "risk_summary",
        "top_comorbidities",
    ):
        tables.update(_run(name, tables))
    return tables


def _check(tables: dict[str, pa.Table]) -> None:
    """The sample must have the shape the README and the tests describe."""
    assert tables["person"].num_rows == COHORT + TOO_EARLY + NO_FOLLOW_UP + NO_DIAGNOSIS
    assert tables["target_cohort"].num_rows == COHORT, tables["target_cohort"].num_rows
    metrics = {row["split"]: row for row in tables["model_metrics"].to_pylist()}
    assert 0.70 <= metrics["test"]["roc_auc"] <= 0.90, metrics["test"]["roc_auc"]


def main() -> None:
    tables = generate()
    _check(tables)
    settings = Settings(sample_dir=REPO / "data" / "sample", output_dir=REPO / "data" / "sample")
    for name, table in tables.items():
        io.write_table(name, table, settings)
    metrics = tables["model_metrics"].to_pylist()
    print(
        f"persons: {tables['person'].num_rows}, cohort: {tables['target_cohort'].num_rows}, "
        f"outcomes: {tables['outcomes'].num_rows}"
    )
    for row in metrics:
        print(f"{row['split']}: {row['notes']} patients, roc_auc {row['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
