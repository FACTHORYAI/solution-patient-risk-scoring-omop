"""Outcome cohort: emergency-room visits inside the time-at-risk window."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from patient_risk_scoring_omop import io
from patient_risk_scoring_omop.jobs._duck import align, run_query
from patient_risk_scoring_omop.schemas import OUTCOMES

INPUTS = ("target_cohort", "visit_occurrence")
OUTPUTS = ("outcomes",)


@dataclass(frozen=True)
class Params:
    """Parameters for the outcome cohort."""

    outcome_concept_id: int = 9203
    min_time_at_risk: int = 7
    max_time_at_risk: int = 365


def run(inputs: dict[str, pa.Table], params: Params) -> dict[str, pa.Table]:
    """Return the cohort members who visited the emergency room in the risk window."""
    if params.min_time_at_risk < 0:
        message = (
            f"Parameter min_time_at_risk is {params.min_time_at_risk}; it must not be negative"
        )
        raise ValueError(message)
    if params.max_time_at_risk <= params.min_time_at_risk:
        message = (
            f"Parameter max_time_at_risk is {params.max_time_at_risk}; "
            f"it must be greater than min_time_at_risk ({params.min_time_at_risk})"
        )
        raise ValueError(message)
    table = run_query(
        io.read_sql("build_outcomes"),
        inputs,
        {
            "outcome_concept_id": params.outcome_concept_id,
            "min_time_at_risk": params.min_time_at_risk,
            "max_time_at_risk": params.max_time_at_risk,
        },
    )
    return {"outcomes": align(table, OUTCOMES)}
