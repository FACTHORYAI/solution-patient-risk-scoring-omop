-- How the predicted risk bands line up with the observed outcome.
SELECT CASE
           WHEN predicted_probability < 0.2 THEN '0.0-0.2'
           WHEN predicted_probability < 0.4 THEN '0.2-0.4'
           WHEN predicted_probability < 0.6 THEN '0.4-0.6'
           WHEN predicted_probability < 0.8 THEN '0.6-0.8'
           ELSE '0.8-1.0'
       END AS risk_band,
       count(*) AS notes,
       avg(predicted_probability) AS mean_probability,
       avg(outcome) AS observed_rate
FROM model_predictions
GROUP BY risk_band
ORDER BY risk_band
