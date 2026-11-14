-- Target cohort: persons with a condition record of the target condition or any
-- descendant, indexed at their first diagnosis, with enough prior observation and
-- enough follow-up to be at risk.
WITH descendants AS (
    SELECT descendant_concept_id AS condition_concept_id
    FROM concept_ancestor
    WHERE ancestor_concept_id = $target_condition_concept_id
),
earliest_onset AS (
    SELECT co.person_id,
           min(co.condition_start_date) AS condition_start_date
    FROM condition_occurrence AS co
    WHERE co.condition_concept_id IN (SELECT condition_concept_id FROM descendants)
    GROUP BY co.person_id
),
observation AS (
    SELECT person_id,
           min(visit_start_date) AS observation_start,
           max(visit_start_date) AS observation_end
    FROM visit_occurrence
    GROUP BY person_id
)
SELECT eo.person_id AS subject_id,
       CAST(eo.condition_start_date - to_days($min_observation_period) AS DATE) AS cohort_start_date,
       eo.condition_start_date AS cohort_end_date
FROM earliest_onset AS eo
JOIN observation AS obs
    ON obs.person_id = eo.person_id
WHERE eo.condition_start_date >= obs.observation_start + to_days($min_observation_period)
  AND eo.condition_start_date + to_days($max_time_at_risk) < obs.observation_end
ORDER BY subject_id
