"""Typed settings, with defaults and loading from a TOML file."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Where the CLI reads sample tables and writes its outputs."""

    sample_dir: Path = Path("data") / "sample"
    output_dir: Path = Path("output")


DEFAULT_SETTINGS_PATH = Path("patient_risk_scoring_omop.toml")


def load_settings(path: Path | None = None) -> Settings:
    """Return Settings from a TOML file, or the defaults when the file is missing."""
    settings_path = path if path is not None else DEFAULT_SETTINGS_PATH
    if not settings_path.is_file():
        return Settings()
    with settings_path.open("rb") as handle:
        values = tomllib.load(handle).get("settings", {})
    unknown = set(values) - set(Settings.__dataclass_fields__)
    if unknown:
        message = f"Unknown settings keys in {settings_path}: {sorted(unknown)}"
        raise ValueError(message)
    return Settings(
        sample_dir=Path(values["sample_dir"]) if "sample_dir" in values else Settings.sample_dir,
        output_dir=Path(values["output_dir"]) if "output_dir" in values else Settings.output_dir,
    )
