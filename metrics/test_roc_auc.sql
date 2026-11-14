-- The model's ROC AUC on the held-out test split.
SELECT roc_auc AS test_roc_auc
FROM model_metrics
WHERE split = 'test'
