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

Predicts which patients of a cohort will have an outcome, here an emergency-room visit within a year of a first heart-failure diagnosis, from the history in the OMOP common data model.

1. **Cohort.** `build_target_cohort` indexes each patient at their first diagnosis of the target condition (or a descendant concept), keeping those with enough observation before it and follow-up after it.
2. **Outcome.** `build_outcomes` finds the emergency-room visits inside the time at risk.
3. **Features.** `build_features` measures age, gender, race, drug history, comorbidity history and prior emergency use from the history before the index date. `build_training_data` joins cohort, outcome and features into one training table.
4. **Model.** `train_risk_model` fits a standardised logistic regression on a stratified train, validation and test split and writes metrics per split, odds ratios per feature and a predicted probability per patient. `risk_summary` groups the predictions into risk bands against the observed rate.

Steps 1 to 3 are DuckDB SQL; step 4 is scikit-learn. No language model is involved.

## How it works in Facthory

- Jobs read tables and write tables. They draw no screens and own no storage; querying, charting, joining with your own data, permissions and audit belong to the platform.
- The agent skill in `skills/` and the starter page in `pages/` describe how to work with the tables and the jobs.
- Sample data in this repository is synthetic. No real patient data.

## What the result could look like

![Mock-up of a Facthory report page titled "Patient risk results", marked "Sample data": four figures (patients in the cohort, observed outcome rate, test ROC AUC, patients in the top risk band), a table of the predicted probability bands against the observed emergency-visit rate, a bar chart of the odds ratio per feature, and a bar chart of the ROC AUC for the training, validation and test splits.](docs/images/patient-risk-results-mockup.png)

A design illustration of a report built from this solution's tables, drawn from this repository's own sample data (generated, with a signal built in). It is a mock-up and not a screenshot of a released feature. The performance shown is on synthetic data and is not evidence about any real population.

## The design in two pictures

These diagrams show the design. They are a concept, not a map of finished platform features.

**Training flow.** The parameters define a target cohort and an outcome cohort from the OMOP clinical data. Drugs and conditions become attributes, attributes and demographics become features, the features are joined into a training dataset with the outcome as the label, and a model is trained on it.

![Sequence diagram "Training a model from OMOP data": the user defines parameters (codes, time at risk, observation period); clinical data yields a target cohort and an outcome cohort; cohort attributes are added for drugs and conditions; features are created for drugs, history and demographics in a feature store; the features are joined into a training dataset and the outcome is added as a label; a classification step in AutoML produces the best model, which goes to a model registry.](docs/images/omop-model-training-flow.png)

This repository covers the steps from the parameters to the training dataset (`build_target_cohort`, `build_outcomes`, `build_features`, `build_training_data`) and trains one scikit-learn logistic-regression model (`train_risk_model`). An AutoML search, a feature store and a model registry are not part of it.

**Observation window and time at risk.** Events before the index date, t = 0, become features; the outcome is looked for after t = 0. These are the observation window and time-at-risk parameters of the cohort jobs.

![Timeline diagram "Observation window and time at risk": visits, drugs, measurements and conditions that happen before t = 0 fall in the observation window and become features; after t = 0 the time at risk runs until its end, and the outcome is looked for in it.](docs/images/observation-window-and-time-at-risk.png)

## Jobs

| Job | What it does | Inputs | Outputs |
| --- | --- | --- | --- |
| `build_target_cohort` | Indexes members at their first diagnosis of the target condition, with enough prior observation and follow-up | `person`, `condition_occurrence`, `concept_ancestor`, `visit_occurrence` | `target_cohort` |
| `build_outcomes` | Finds the emergency-room visits inside the time-at-risk window | `target_cohort`, `visit_occurrence` | `outcomes` |
| `build_features` | Age, gender, race, drug history, comorbidity history and prior emergency use, measured before the index date | `target_cohort`, `person`, `condition_era`, `drug_exposure`, `visit_occurrence` | `person_features` |
| `build_training_data` | Joins cohort, outcome and features into one training table | `target_cohort`, `outcomes`, `person_features` | `training_data` |
| `train_risk_model` | Trains a standardised logistic-regression risk model on a stratified train/validation/test split; race is left out of the model unless `include_race` is set | `training_data` | `model_metrics`, `feature_importance`, `model_predictions` |
| `risk_summary` | Predicted risk bands against the observed outcome rate | `model_predictions` | `risk_bands` |
| `top_comorbidities` | The most frequent conditions in the cohort history | `condition_era`, `target_cohort`, `concept` | `top_comorbidities` |

Every job is a plain function from named PyArrow tables to named tables: cohort and feature queries run as DuckDB SQL (`sql/`), the model is a scikit-learn pipeline, and its metrics, coefficients and predictions are tables. Named measures live in `metrics/`.

## Data

The sample tables in `data/sample/` are a synthetic OMOP 5.3 extract generated by `data/generate_sample.py` (CC0-1.0): 900 invented persons, of whom 650 form the cohort, reduced to the columns the pipeline reads, plus the derived cohort, feature and model tables. The generator builds a signal into the outcome so the model has something to learn. See [data/README.md](data/README.md) for one row per file with origin and licence. No real patient data.

## Run it locally

```
uv sync
uv run python -m patient_risk_scoring_omop build_target_cohort
uv run python -m patient_risk_scoring_omop train_risk_model --test-fraction 0.2 --validation-fraction 0.2
uv run python -m patient_risk_scoring_omop train_risk_model --include-race
uv run python -m patient_risk_scoring_omop risk_summary
uv run pytest
```

`uv run python -m patient_risk_scoring_omop --help` lists every job; each job has its own `--help`. The cohort jobs take the OHDSI patient-level prediction parameters (target and outcome concept ids, observation and time-at-risk windows, lookback days); the model takes the split fractions, the decision threshold and the random seed, so a run is reproducible.

The risk model is a statistical association over historical data, not a clinical decision: `model_metrics` reports performance per split, `feature_importance` ranks the features by coefficient with odds ratios, and `risk_bands` compares predicted probability with the observed outcome rate.

A run is reproducible: the model takes the split fractions, the decision threshold and the random seed. Race is not a model feature unless `--include-race` is given; it stays in `training_data` so a fairness check is possible.

## Open points

- **The performance shown is on synthetic data with a built-in signal.** The generator draws the outcome from a logistic function of age at diagnosis, comorbidities, prior emergency visits and warfarin exposure, plus noise. A test ROC AUC of about 0.78 here says nothing about any real population or model.
- **The model predicts an individual patient's chance of an emergency visit.** It is for research. Its use on real patients needs a review of its regulatory weight before anyone acts on a score.
- **Race is left out of the model by default, and gender is still a feature.** A model that uses gender or other demographics needs a fairness review: compare `model_predictions` across groups before relying on it.
- **Only one target condition and one outcome are parameterised** (concept ids and windows), not a general cohort definition language.
- **AutoML, a feature store and a model registry are not included.** One scikit-learn logistic regression is trained; the diagrams show the wider design.
- **At OMOP scale the cohort and feature SQL must run in the analytics engine,** and only the training table goes to the Python step: a code run takes at most 32 MiB of input, 120 seconds, 512 MiB and one CPU.
- **Installing into Facthory is not available yet.** The code is verified on the sample data on a laptop only.
- **The Dockerfile has not been built or verified.**
- **DuckDB 2.x only.** The code uses the DuckDB 2.0 Arrow API (`to_arrow_table`).
- No orchestrator or DAG files: Facthory, or a scheduler, calls the jobs.

## Not for clinical use

This is software for research and analytics. It is not a medical device and must not be used to diagnose or treat a patient. Results can be wrong; a qualified person reviews them.

## Disclaimers

Bandpey GmbH (“Facthory”) does not dispense medical, diagnosis, or treatment advice. This Solution (“tool”) is for informational purposes only and may not be used as a substitute for professional medical advice, treatment, or diagnosis. This tool may not be used within Facthory to process Protected Health Information (“PHI”) as defined in the Health Insurance Portability and Accountability Act of 1996, unless you have executed with Facthory a contract that allows for processing PHI, an accompanying Business Associate Agreement (BAA), and are running this tool within a HIPAA Account.

## Licence

Apache-2.0. See [LICENSE](LICENSE). "Facthory" and the Facthory logo are trademarks of their owner and are not covered by that licence. This includes the logo shown in the images under `docs/images/`.

