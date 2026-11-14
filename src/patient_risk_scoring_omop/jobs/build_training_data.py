"""The training dataset: cohort members with their outcome label and features."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from patient_risk_scoring_omop import io
from patient_risk_scoring_omop.jobs._duck import align, run_query
from patient_risk_scoring_omop.schemas import TRAINING_DATA

INPUTS = ("target_cohort", "outcomes", "person_features")
OUTPUTS = ("training_data",)


@dataclass(frozen=True)
class Params:
    """Parameters for the training dataset."""


def run(inputs: dict[str, pa.Table], params: Params) -> dict[str, pa.Table]:
    """Return one row per cohort member with the binary outcome and the feature columns."""
    table = run_query(io.read_sql("build_training_data"), inputs)
    return {"training_data": align(table, TRAINING_DATA)}
