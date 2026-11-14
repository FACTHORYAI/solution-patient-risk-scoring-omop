"""The only module that reads or writes files."""

from __future__ import annotations

import csv
import logging
from importlib import resources
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from patient_risk_scoring_omop.config import Settings

logger = logging.getLogger(__name__)


def read_table(name: str, settings: Settings | None = None) -> pa.Table:
    """Read a sample table by name (Parquet or CSV) from the sample directory."""
    settings = settings or Settings()
    sample_dir = Path(settings.sample_dir)
    for suffix in (".parquet", ".csv"):
        path = sample_dir / f"{name}{suffix}"
        if path.is_file():
            return pq.read_table(path) if suffix == ".parquet" else _read_csv(path)
    message = f"Sample table {name!r} not found in {sample_dir}"
    raise FileNotFoundError(message)


def write_table(name: str, table: pa.Table, settings: Settings | None = None) -> Path:
    """Write a table as Parquet into the output directory and return its path."""
    settings = settings or Settings()
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.parquet"
    pq.write_table(table, path)
    logger.info("Wrote %s (%d rows)", path, table.num_rows)
    return path


def read_sql(name: str) -> str:
    """Load a DuckDB SQL query by name from the package's sql/ directory."""
    package = resources.files("patient_risk_scoring_omop")
    return package.joinpath("sql").joinpath(f"{name}.sql").read_text(encoding="utf-8")


def _read_csv(path: Path) -> pa.Table:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        message = f"CSV file {path} has no rows"
        raise ValueError(message)
    columns: dict[str, list[str]] = {key: [] for key in rows[0]}
    for row in rows:
        for key, value in row.items():
            columns[key].append(value)
    return pa.table(columns)
