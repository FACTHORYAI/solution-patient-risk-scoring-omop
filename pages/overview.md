# Patient Risk Scoring (OMOP)

Score patient risk from OMOP condition history.

## Start here

1. Look at the sample tables in `data/sample`: an invented OMOP 5.3 extract of 900 persons (650 in the cohort), made by `data/generate_sample.py`, and the derived cohort, feature and model tables.
2. Follow the OHDSI patient-level prediction design: `build_target_cohort` indexes members at their first heart-failure diagnosis, `build_outcomes` finds emergency-room visits in the risk window, `build_features` measures history before the index date, and `build_training_data` joins them.
3. Train and read the model:

| Question | Job |
| --- | --- |
| Who is in the cohort? | `build_target_cohort` |
| Who experienced the outcome? | `build_outcomes` |
| What are the features? | `build_features` |
| What predicts risk? | `train_risk_model` (`feature_importance`) |
| How well does the model perform? | `train_risk_model` (`model_metrics`) |
| How do risk bands relate to observed outcomes? | `risk_summary` |
| What conditions are most common in the cohort? | `top_comorbidities` |

Named measures for the outcome rate and the test ROC AUC live in `metrics/`.

## Notes

- The sample data is synthetic; real patient data never enters this project.
- The risk model is a logistic regression: it shows association, not causation. Race is left out of the model by default. The sample has a signal built in, so its performance says nothing about any real population.
- Results are for research and analytics, not clinical decisions.
