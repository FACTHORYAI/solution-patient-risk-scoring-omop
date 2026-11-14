# Sample data

Every file below is synthetic, generated for this repository from a seeded generator
that writes OMOP 5.3-shaped tables; no real patient data. Licence: CC0-1.0. The folder
is about 90 KB in total, far under the 5 MB per file and 20 MB per folder limits.

The raw tables follow the OMOP CDM 5.3 column names, reduced to the columns this
pipeline reads:

| File | What it is | Origin | Licence |
| --- | --- | --- | --- |
| `person.parquet` | 240 persons: gender, year of birth, race | Seeded generator | CC0-1.0 |
| `condition_occurrence.parquet` | Condition records, including the index heart-failure diagnosis | Seeded generator | CC0-1.0 |
| `concept_ancestor.parquet` | The heart-failure concept and its descendant | Seeded generator | CC0-1.0 |
| `visit_occurrence.parquet` | Outpatient observation visits and emergency-room visits | Seeded generator | CC0-1.0 |
| `condition_era.parquet` | Condition eras with occurrence counts, for the comorbidity history | Seeded generator | CC0-1.0 |
| `drug_exposure.parquet` | Warfarin and acetaminophen exposures, for the drug history | Seeded generator | CC0-1.0 |
| `concept.parquet` | Concept names for the concepts used above | Seeded generator | CC0-1.0 |
| `target_cohort.parquet` | Cohort members indexed at first heart-failure diagnosis | `build_target_cohort` | CC0-1.0 |
| `outcomes.parquet` | Members with an emergency-room visit in the risk window | `build_outcomes` | CC0-1.0 |
| `person_features.parquet` | Age, gender, race, drug history, comorbidity history, prior ER use | `build_features` | CC0-1.0 |
| `training_data.parquet` | One row per member: outcome label plus features | `build_training_data` | CC0-1.0 |
| `model_metrics.parquet` | Accuracy, precision, recall, F1 and ROC AUC per split | `train_risk_model` | CC0-1.0 |
| `feature_importance.parquet` | Logistic-regression coefficient and odds ratio per feature | `train_risk_model` | CC0-1.0 |
| `model_predictions.parquet` | Predicted probability and predicted outcome per member | `train_risk_model` | CC0-1.0 |
| `risk_bands.parquet` | Predicted risk bands against the observed outcome rate | `risk_summary` | CC0-1.0 |
| `top_comorbidities.parquet` | The most frequent conditions in the cohort history | `top_comorbidities` | CC0-1.0 |

The generator builds in a real signal: the probability of an emergency visit rises with
age at diagnosis, the number of comorbidities, prior emergency-room use and warfarin
exposure, so the model has something to learn and the risk bands trend upward.

The original solution's dataset was **not** republished: it is a Synthea-derived OMOP
extract hosted under the Databricks licence, which does not permit reuse outside the
Databricks platform. This repository ships its own synthetic OMOP extract instead; a
workspace run points the same jobs at a real OMOP CDM through Facthory's own storage.
