"""The jobs of this solution: plain functions from named tables to named tables."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from patient_risk_scoring_omop.jobs import (
    build_features,
    build_outcomes,
    build_target_cohort,
    build_training_data,
    risk_summary,
    top_comorbidities,
    train_risk_model,
)


@dataclass(frozen=True)
class JobSpec:
    """One job: its callable, its parameter class and its input and output tables."""

    name: str
    description: str
    run: Callable[..., dict[str, pa.Table]]
    params_cls: type
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


REGISTRY: dict[str, JobSpec] = {
    spec.name: spec
    for spec in (
        JobSpec(
            name="build_target_cohort",
            description="Target cohort: persons newly diagnosed with the target condition.",
            run=build_target_cohort.run,
            params_cls=build_target_cohort.Params,
            inputs=build_target_cohort.INPUTS,
            outputs=build_target_cohort.OUTPUTS,
        ),
        JobSpec(
            name="build_outcomes",
            description="Outcome cohort: emergency-room visits inside the time-at-risk window.",
            run=build_outcomes.run,
            params_cls=build_outcomes.Params,
            inputs=build_outcomes.INPUTS,
            outputs=build_outcomes.OUTPUTS,
        ),
        JobSpec(
            name="build_features",
            description="Person features from demographics, condition, drug and visit history.",
            run=build_features.run,
            params_cls=build_features.Params,
            inputs=build_features.INPUTS,
            outputs=build_features.OUTPUTS,
        ),
        JobSpec(
            name="build_training_data",
            description="The training dataset: cohort members with their outcome and features.",
            run=build_training_data.run,
            params_cls=build_training_data.Params,
            inputs=build_training_data.INPUTS,
            outputs=build_training_data.OUTPUTS,
        ),
        JobSpec(
            name="train_risk_model",
            description="Train and evaluate a logistic-regression risk model on the training data.",
            run=train_risk_model.run,
            params_cls=train_risk_model.Params,
            inputs=train_risk_model.INPUTS,
            outputs=train_risk_model.OUTPUTS,
        ),
        JobSpec(
            name="risk_summary",
            description="How the predicted risk bands line up with the observed outcome.",
            run=risk_summary.run,
            params_cls=risk_summary.Params,
            inputs=risk_summary.INPUTS,
            outputs=risk_summary.OUTPUTS,
        ),
        JobSpec(
            name="top_comorbidities",
            description="The conditions most often present in the cohort's history.",
            run=top_comorbidities.run,
            params_cls=top_comorbidities.Params,
            inputs=top_comorbidities.INPUTS,
            outputs=top_comorbidities.OUTPUTS,
        ),
    )
}


def param_values(spec: JobSpec, args: Any) -> Any:
    """Build the job's Params instance from parsed command-line arguments."""
    values = {
        field.name: getattr(args, field.name)
        for field in spec.params_cls.__dataclass_fields__.values()
    }
    return spec.params_cls(**values)
