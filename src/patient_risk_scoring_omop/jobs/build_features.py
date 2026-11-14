"""Person-level features from demographics, condition, drug and visit history."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from patient_risk_scoring_omop import io
from patient_risk_scoring_omop.jobs._duck import align, run_query
from patient_risk_scoring_omop.schemas import PERSON_FEATURES

INPUTS = ("target_cohort", "person", "condition_era", "drug_exposure", "visit_occurrence")
OUTPUTS = ("person_features",)


@dataclass(frozen=True)
class Params:
    """Parameters for the person features."""

    drug1_concept_id: int = 40163554
    drug2_concept_id: int = 40221901
    outcome_concept_id: int = 9203
    lookback_days: int = 1825


def run(inputs: dict[str, pa.Table], params: Params) -> dict[str, pa.Table]:
    """Return one feature row per cohort member, measured before the cohort start."""
    if params.lookback_days < 1:
        message = f"Parameter lookback_days is {params.lookback_days}; it must be at least 1"
        raise ValueError(message)
    table = run_query(
        io.read_sql("build_features"),
        inputs,
        {
            "drug1_concept_id": params.drug1_concept_id,
            "drug2_concept_id": params.drug2_concept_id,
            "outcome_concept_id": params.outcome_concept_id,
            "lookback_days": params.lookback_days,
        },
    )
    return {"person_features": align(table, PERSON_FEATURES)}
