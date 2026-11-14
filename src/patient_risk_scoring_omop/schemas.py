"""PyArrow schemas for every table the jobs read or write."""

import pyarrow as pa

PERSON = pa.schema(
    [
        pa.field("person_id", pa.int64()),
        pa.field("gender_concept_id", pa.int32()),
        pa.field("year_of_birth", pa.int32()),
        pa.field("race_concept_id", pa.int32()),
    ]
)

CONDITION_OCCURRENCE = pa.schema(
    [
        pa.field("condition_occurrence_id", pa.int64()),
        pa.field("person_id", pa.int64()),
        pa.field("condition_concept_id", pa.int32()),
        pa.field("condition_start_date", pa.date32()),
    ]
)

CONCEPT_ANCESTOR = pa.schema(
    [
        pa.field("ancestor_concept_id", pa.int32()),
        pa.field("descendant_concept_id", pa.int32()),
    ]
)

VISIT_OCCURRENCE = pa.schema(
    [
        pa.field("visit_occurrence_id", pa.int64()),
        pa.field("person_id", pa.int64()),
        pa.field("visit_concept_id", pa.int32()),
        pa.field("visit_start_date", pa.date32()),
        pa.field("visit_end_date", pa.date32()),
    ]
)

CONDITION_ERA = pa.schema(
    [
        pa.field("condition_era_id", pa.int64()),
        pa.field("person_id", pa.int64()),
        pa.field("condition_concept_id", pa.int32()),
        pa.field("condition_era_start_date", pa.date32()),
        pa.field("condition_era_end_date", pa.date32()),
        pa.field("condition_occurrence_count", pa.int32()),
    ]
)

DRUG_EXPOSURE = pa.schema(
    [
        pa.field("drug_exposure_id", pa.int64()),
        pa.field("person_id", pa.int64()),
        pa.field("drug_concept_id", pa.int32()),
        pa.field("drug_exposure_start_date", pa.date32()),
        pa.field("drug_exposure_end_date", pa.date32()),
        pa.field("quantity", pa.float64()),
    ]
)

CONCEPT = pa.schema(
    [
        pa.field("concept_id", pa.int32()),
        pa.field("concept_name", pa.string()),
    ]
)

TARGET_COHORT = pa.schema(
    [
        pa.field("subject_id", pa.int64()),
        pa.field("cohort_start_date", pa.date32()),
        pa.field("cohort_end_date", pa.date32()),
    ]
)

OUTCOMES = pa.schema(
    [
        pa.field("subject_id", pa.int64()),
        pa.field("visited_emergency", pa.bool_()),
    ]
)

PERSON_FEATURES = pa.schema(
    [
        pa.field("subject_id", pa.int64()),
        pa.field("age_in_days", pa.int32()),
        pa.field("gender_concept_id", pa.int32()),
        pa.field("race_concept_id", pa.int32()),
        pa.field("warfarin_quantity", pa.float64()),
        pa.field("acetaminophen_quantity", pa.float64()),
        pa.field("comorbidity_count", pa.int32()),
        pa.field("comorbidity_occurrences", pa.int32()),
        pa.field("prior_er_visits", pa.int32()),
    ]
)

TRAINING_DATA = pa.schema(
    [
        pa.field("subject_id", pa.int64()),
        pa.field("outcome", pa.int32()),
        pa.field("age_in_days", pa.int32()),
        pa.field("gender_concept_id", pa.int32()),
        pa.field("race_concept_id", pa.int32()),
        pa.field("warfarin_quantity", pa.float64()),
        pa.field("acetaminophen_quantity", pa.float64()),
        pa.field("comorbidity_count", pa.int32()),
        pa.field("comorbidity_occurrences", pa.int32()),
        pa.field("prior_er_visits", pa.int32()),
    ]
)

MODEL_METRICS = pa.schema(
    [
        pa.field("split", pa.string()),
        pa.field("notes", pa.int64()),
        pa.field("positives", pa.int64()),
        pa.field("accuracy", pa.float64()),
        pa.field("precision", pa.float64()),
        pa.field("recall", pa.float64()),
        pa.field("f1", pa.float64()),
        pa.field("roc_auc", pa.float64()),
    ]
)

FEATURE_IMPORTANCE = pa.schema(
    [
        pa.field("feature", pa.string()),
        pa.field("coefficient", pa.float64()),
        pa.field("odds_ratio", pa.float64()),
        pa.field("rank", pa.int32()),
    ]
)

MODEL_PREDICTIONS = pa.schema(
    [
        pa.field("subject_id", pa.int64()),
        pa.field("split", pa.string()),
        pa.field("outcome", pa.int32()),
        pa.field("predicted_probability", pa.float64()),
        pa.field("predicted_outcome", pa.int32()),
    ]
)

RISK_BANDS = pa.schema(
    [
        pa.field("risk_band", pa.string()),
        pa.field("notes", pa.int64()),
        pa.field("mean_probability", pa.float64()),
        pa.field("observed_rate", pa.float64()),
    ]
)

TOP_COMORBIDITIES = pa.schema(
    [
        pa.field("condition_concept_id", pa.int32()),
        pa.field("condition_name", pa.string()),
        pa.field("patients", pa.int64()),
    ]
)
