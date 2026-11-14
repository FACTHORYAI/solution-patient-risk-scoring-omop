# Patient Risk Scoring (OMOP) — agent instructions

You help members score patient risk from OMOP condition history: build the cohort, build the features, train and read the risk model, and judge how well it performs.

## What the project provides

- Sample tables in `data/sample` (an invented OMOP 5.3 extract of 900 persons, 650 in the cohort): the raw `person`, `condition_occurrence`, `concept_ancestor`, `visit_occurrence`, `condition_era`, `drug_exposure` and `concept` tables, plus the derived `target_cohort`, `outcomes`, `person_features`, `training_data`, `model_metrics`, `feature_importance`, `model_predictions`, `risk_bands` and `top_comorbidities`.
- Jobs under `src/patient_risk_scoring_omop/jobs`: four cohort and feature jobs (SQL), the risk model (scikit-learn logistic regression), and two summaries.

## How to answer a question

1. Classify what the member asks: who is in the cohort (`target_cohort`), how many experienced the outcome (`outcomes`, `metrics/outcome_rate.sql`), what predicts risk (`feature_importance`), how well the model separates the classes (`model_metrics`, `metrics/test_roc_auc.sql`), or how risk bands relate to observed outcomes (`risk_bands`).
2. Check the parameters with `uv run python -m patient_risk_scoring_omop <job> --help`; the cohort jobs take concept ids and windows (`min_observation_period`, `min_time_at_risk`, `max_time_at_risk`, `lookback_days`), the model takes split fractions, a threshold, a random seed and `include_race` (off by default: race stays in `training_data` but is not a model feature).
3. Run the job, or answer directly from the tables: `training_data` holds one row per cohort member with the outcome label and features; `feature_importance` ranks features by coefficient with odds ratios; `risk_bands` compares predicted probability with the observed rate.
4. Report results as a table with the counts behind them. Never present a risk score as a clinical decision: the model is a statistical association over historical data.

## Data care

- The sample data is synthetic; real patient data never enters this project.
- The model is a logistic regression over cohort features: it shows association, not causation. The sample has a signal built in by its generator, so its test ROC AUC of about 0.78 says nothing about any real population; say so when you report performance.
- The model predicts an individual patient's chance of an emergency visit: for research only, and gender or other demographics in a model need a fairness review (compare `model_predictions` across groups).
- Results are for research and analytics, not clinical decisions.
