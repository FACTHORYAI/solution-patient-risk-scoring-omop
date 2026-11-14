"""The command line: one sub-command per job."""

from __future__ import annotations

import argparse
import logging
from dataclasses import MISSING, fields
from typing import Any

from patient_risk_scoring_omop import io
from patient_risk_scoring_omop.config import load_settings
from patient_risk_scoring_omop.jobs import REGISTRY, JobSpec, param_values

logger = logging.getLogger(__name__)

_TYPE_MAP = {"str": str, "int": int, "float": float, "bool": bool}


def main(argv: list[str] | None = None) -> int:
    """Parse the command line, run one job on the sample data and write its outputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    spec: JobSpec = args.job_spec
    settings = load_settings()
    inputs = {name: io.read_table(name, settings) for name in spec.inputs}
    outputs = spec.run(inputs=inputs, params=param_values(spec, args))
    for name, table in outputs.items():
        io.write_table(name, table, settings)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="patient_risk_scoring_omop",
        description="Score patient risk from OMOP condition history.",
    )
    subparsers = parser.add_subparsers(dest="job", required=True)
    for name, spec in REGISTRY.items():
        sub = subparsers.add_parser(name, help=spec.description)
        _add_params(sub, spec.params_cls)
        sub.set_defaults(job_spec=spec)
    return parser


def _add_params(sub: argparse.ArgumentParser, params_cls: type) -> None:
    for field in fields(params_cls):
        flag = f"--{field.name.replace('_', '-')}"
        argument_type = _TYPE_MAP.get(str(field.type), str)
        if argument_type is bool:
            sub.add_argument(
                flag, dest=field.name, action=argparse.BooleanOptionalAction, default=field.default
            )
            continue
        options: dict[str, Any] = {"type": argument_type, "dest": field.name}
        if field.default is MISSING:
            options["required"] = True
        else:
            options["default"] = field.default
        sub.add_argument(flag, **options)
