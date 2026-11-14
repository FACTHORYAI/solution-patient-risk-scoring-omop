"""The conditions most often present in the cohort's history."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from patient_risk_scoring_omop import io
from patient_risk_scoring_omop.jobs._duck import align, run_query
from patient_risk_scoring_omop.schemas import TOP_COMORBIDITIES

INPUTS = ("condition_era", "target_cohort", "concept")
OUTPUTS = ("top_comorbidities",)


@dataclass(frozen=True)
class Params:
    """Parameters for the comorbidity summary."""

    lookback_days: int = 1825
    limit: int = 5


def run(inputs: dict[str, pa.Table], params: Params) -> dict[str, pa.Table]:
    """Return the most frequent conditions in the cohort history, with patient counts."""
    if params.lookback_days < 1:
        message = f"Parameter lookback_days is {params.lookback_days}; it must be at least 1"
        raise ValueError(message)
    if params.limit < 1:
        message = f"Parameter limit is {params.limit}; it must be at least 1"
        raise ValueError(message)
    table = run_query(
        io.read_sql("top_comorbidities"),
        inputs,
        {"lookback_days": params.lookback_days, "limit": params.limit},
    )
    return {"top_comorbidities": align(table, TOP_COMORBIDITIES)}
