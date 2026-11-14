-- Outcome: the person visited the emergency room within the time-at-risk window
-- that starts $min_time_at_risk days after the cohort start.
SELECT DISTINCT
    tc.subject_id,
    true AS visited_emergency
FROM target_cohort AS tc
JOIN visit_occurrence AS vo
    ON vo.person_id = tc.subject_id
WHERE vo.visit_concept_id = $outcome_concept_id
  AND vo.visit_start_date >= tc.cohort_end_date + to_days($min_time_at_risk)
  AND vo.visit_start_date < tc.cohort_end_date + to_days($max_time_at_risk)
ORDER BY tc.subject_id
