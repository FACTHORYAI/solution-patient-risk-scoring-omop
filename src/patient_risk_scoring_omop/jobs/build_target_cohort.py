"""Target cohort: persons newly diagnosed with the target condition."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from patient_risk_scoring_omop import io
from patient_risk_scoring_omop.jobs._duck import align, run_query
from patient_risk_scoring_omop.schemas import TARGET_COHORT

INPUTS = ("person", "condition_occurrence", "concept_ancestor", "visit_occurrence")
OUTPUTS = ("target_cohort",)


@dataclass(frozen=True)
class Params:
    """Parameters for the target cohort."""

    target_condition_concept_id: int = 4229440
    min_observation_period: int = 1095
    max_time_at_risk: int = 365


def run(inputs: dict[str, pa.Table], params: Params) -> dict[str, pa.Table]:
    """Return one row per person indexed at their first diagnosis of the condition."""
    if params.min_observation_period < 0:
        message = (
            f"Parameter min_observation_period is {params.min_observation_period}; "
            "it must not be negative"
        )
        raise ValueError(message)
    if params.max_time_at_risk < 1:
        message = f"Parameter max_time_at_risk is {params.max_time_at_risk}; it must be at least 1"
        raise ValueError(message)
    table = run_query(
        io.read_sql("build_target_cohort"),
        inputs,
        {
            "target_condition_concept_id": params.target_condition_concept_id,
            "min_observation_period": params.min_observation_period,
            "max_time_at_risk": params.max_time_at_risk,
        },
    )
    return {"target_cohort": align(table, TARGET_COHORT)}
