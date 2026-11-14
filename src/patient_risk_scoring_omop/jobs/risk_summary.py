"""How the predicted risk bands line up with the observed outcome."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from patient_risk_scoring_omop import io
from patient_risk_scoring_omop.jobs._duck import align, run_query
from patient_risk_scoring_omop.schemas import RISK_BANDS

INPUTS = ("model_predictions",)
OUTPUTS = ("risk_bands",)


@dataclass(frozen=True)
class Params:
    """Parameters for the risk-band summary."""


def run(inputs: dict[str, pa.Table], params: Params) -> dict[str, pa.Table]:
    """Return note counts, mean predicted probability and observed rate per risk band."""
    table = run_query(io.read_sql("risk_bands"), inputs)
    return {"risk_bands": align(table, RISK_BANDS)}
