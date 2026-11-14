-- The training dataset: one row per cohort member with the outcome label and features.
SELECT tc.subject_id,
       CAST(coalesce(o.visited_emergency, false) AS INTEGER) AS outcome,
       pf.age_in_days,
       pf.gender_concept_id,
       pf.race_concept_id,
       pf.warfarin_quantity,
       pf.acetaminophen_quantity,
       pf.comorbidity_count,
       pf.comorbidity_occurrences,
       pf.prior_er_visits
FROM target_cohort AS tc
JOIN person_features AS pf
    ON pf.subject_id = tc.subject_id
LEFT JOIN outcomes AS o
    ON o.subject_id = tc.subject_id
ORDER BY tc.subject_id
