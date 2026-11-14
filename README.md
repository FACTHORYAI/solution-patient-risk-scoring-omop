<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FACTHORYAI/.github/main/brand/word-mark-only-mono.svg">
    <img alt="Facthory" src="https://raw.githubusercontent.com/FACTHORYAI/.github/main/brand/word-mark-only.svg" width="360">
  </picture>
</p>

# Patient Risk Scoring on OMOP

Predict a patient's risk of an outcome from condition, medication and encounter history in the OMOP common data model, in Facthory.

![Licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue) ![Runs on Facthory](https://img.shields.io/badge/runs%20on-Facthory-1f6feb) ![Status: install coming](https://img.shields.io/badge/status-install%20coming-lightgrey)

> **Install coming.** This is a Facthory solution: its data, analysis code, agent skill and starting pages load into a Facthory workspace as a ready project. Installing from this page is not available yet; this repository is being prepared for it.

## What it does

Predict a patient's risk of an outcome from condition, medication and encounter history in the OMOP common data model, in Facthory.

## How it will work in Facthory

- You install it into a workspace and get a project: sample data, the analysis, the agent's instructions and starting pages.
- Its results are tables and pages in your workspace, so you can query them, chart them and ask the agent about them, inside your permissions.
- Sample data in this repository is synthetic or openly licensed. No real patient data.

## Not for clinical use

This is software for research and analytics. It is not a medical device and must not be used to diagnose or treat a patient. Results can be wrong; a qualified person reviews them.

## Licence

Apache-2.0. See [LICENSE](LICENSE). "Facthory" and the Facthory logo are trademarks of their owner and are not covered by that licence.

## Jobs

| Job | What it does | Inputs | Outputs |
| --- | --- | --- | --- |
| `build_target_cohort` | Indexes members at their first diagnosis of the target condition, with enough prior observation and follow-up | `person`, `condition_occurrence`, `concept_ancestor`, `visit_occurrence` | `target_cohort` |
| `build_outcomes` | Finds the emergency-room visits inside the time-at-risk window | `target_cohort`, `visit_occurrence` | `outcomes` |
| `build_features` | Age, gender, race, drug history, comorbidity history and prior emergency use, measured before the index date | `target_cohort`, `person`, `condition_era`, `drug_exposure`, `visit_occurrence` | `person_features` |
| `build_training_data` | Joins cohort, outcome and features into one training table | `target_cohort`, `outcomes`, `person_features` | `training_data` |
| `train_risk_model` | Trains a standardised logistic-regression risk model on a stratified train/validation/test split | `training_data` | `model_metrics`, `feature_importance`, `model_predictions` |
| `risk_summary` | Predicted risk bands against the observed outcome rate | `model_predictions` | `risk_bands` |
| `top_comorbidities` | The most frequent conditions in the cohort history | `condition_era`, `target_cohort`, `concept` | `top_comorbidities` |

Every job is a plain function from named PyArrow tables to named tables: cohort and feature queries run as DuckDB SQL (`sql/`), the model is a scikit-learn pipeline, and its metrics, coefficients and predictions are tables. Named measures live in `metrics/`.

## Data

The sample tables in `data/sample/` are a synthetic OMOP 5.3 extract generated for this repository (CC0-1.0): 240 persons, reduced to the columns the pipeline reads, plus the derived cohort, feature and model tables. See [data/README.md](data/README.md) for one row per file with origin and licence. No real patient data.

## Run it locally

```
uv sync
uv run python -m patient_risk_scoring_omop build_target_cohort
uv run python -m patient_risk_scoring_omop train_risk_model --test-fraction 0.2 --validation-fraction 0.2
uv run python -m patient_risk_scoring_omop risk_summary
uv run pytest
```

`uv run python -m patient_risk_scoring_omop --help` lists every job; each job has its own `--help`. The cohort jobs take the OHDSI patient-level prediction parameters (target and outcome concept ids, observation and time-at-risk windows, lookback days); the model takes the split fractions, the decision threshold and the random seed, so a run is reproducible.

The risk model is a statistical association over historical data, not a clinical decision: `model_metrics` reports performance per split, `feature_importance` ranks the features by coefficient with odds ratios, and `risk_bands` compares predicted probability with the observed outcome rate.

