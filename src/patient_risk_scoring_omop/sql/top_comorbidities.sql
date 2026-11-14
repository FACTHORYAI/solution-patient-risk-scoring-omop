-- The conditions most often present in the cohort's history before the index date
-- (cohort_end_date), within $lookback_days of it.
SELECT ce.condition_concept_id,
       coalesce(max(c.concept_name), '') AS condition_name,
       count(DISTINCT ce.person_id) AS patients
FROM condition_era AS ce
JOIN target_cohort AS tc
    ON tc.subject_id = ce.person_id
LEFT JOIN concept AS c
    ON c.concept_id = ce.condition_concept_id
WHERE ce.condition_era_start_date < tc.cohort_end_date
  AND ce.condition_era_start_date >= tc.cohort_end_date - to_days($lookback_days)
GROUP BY ce.condition_concept_id
ORDER BY patients DESC, ce.condition_concept_id
LIMIT $limit
