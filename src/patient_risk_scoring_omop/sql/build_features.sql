-- Person-level features: demographics, drug exposure history, condition history and
-- prior emergency-room use, measured before the index date (cohort_end_date) and, for
-- the drug and condition history, within $lookback_days of it.
WITH cohort AS (
    SELECT subject_id, cohort_start_date, cohort_end_date FROM target_cohort
),
demographics AS (
    SELECT c.subject_id,
           CAST(date_diff('day', make_date(p.year_of_birth, 1, 1), c.cohort_end_date) AS INTEGER)
               AS age_in_days,
           p.gender_concept_id,
           p.race_concept_id
    FROM cohort AS c
    JOIN person AS p ON p.person_id = c.subject_id
),
condition_history AS (
    SELECT c.subject_id,
           count(DISTINCT ce.condition_concept_id) AS comorbidity_count,
           sum(ce.condition_occurrence_count) AS comorbidity_occurrences
    FROM cohort AS c
    JOIN condition_era AS ce ON ce.person_id = c.subject_id
    WHERE ce.condition_era_start_date >= c.cohort_end_date - to_days($lookback_days)
      AND ce.condition_era_start_date < c.cohort_end_date
    GROUP BY c.subject_id
),
drug_history AS (
    SELECT c.subject_id,
           sum(CASE WHEN de.drug_concept_id = $drug1_concept_id THEN de.quantity ELSE 0 END)
               AS warfarin_quantity,
           sum(CASE WHEN de.drug_concept_id = $drug2_concept_id THEN de.quantity ELSE 0 END)
               AS acetaminophen_quantity
    FROM cohort AS c
    JOIN drug_exposure AS de ON de.person_id = c.subject_id
    WHERE de.drug_concept_id IN ($drug1_concept_id, $drug2_concept_id)
      AND de.drug_exposure_start_date >= c.cohort_end_date - to_days($lookback_days)
      AND de.drug_exposure_start_date < c.cohort_end_date
    GROUP BY c.subject_id
),
prior_emergency AS (
    SELECT c.subject_id, count(*) AS prior_er_visits
    FROM cohort AS c
    JOIN visit_occurrence AS vo ON vo.person_id = c.subject_id
    WHERE vo.visit_concept_id = $outcome_concept_id
      AND vo.visit_start_date < c.cohort_end_date
    GROUP BY c.subject_id
)
SELECT d.subject_id,
       d.age_in_days,
       d.gender_concept_id,
       d.race_concept_id,
       coalesce(dh.warfarin_quantity, 0.0) AS warfarin_quantity,
       coalesce(dh.acetaminophen_quantity, 0.0) AS acetaminophen_quantity,
       coalesce(ch.comorbidity_count, 0) AS comorbidity_count,
       coalesce(ch.comorbidity_occurrences, 0) AS comorbidity_occurrences,
       coalesce(pe.prior_er_visits, 0) AS prior_er_visits
FROM demographics AS d
LEFT JOIN drug_history AS dh ON dh.subject_id = d.subject_id
LEFT JOIN condition_history AS ch ON ch.subject_id = d.subject_id
LEFT JOIN prior_emergency AS pe ON pe.subject_id = d.subject_id
ORDER BY d.subject_id
