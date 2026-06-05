### SHAP Interpretation Summary
Top 5 most influential features for mortality risk:

- 1. **age** (mean |SHAP| = 0.4602, share = 28.5%)
- 2. **resp_rate** (mean |SHAP| = 0.4422, share = 27.4%)
- 3. **heart_rate** (mean |SHAP| = 0.3148, share = 19.5%)
- 4. **dbp** (mean |SHAP| = 0.2023, share = 12.5%)
- 5. **temperature** (mean |SHAP| = 0.1929, share = 12.0%)

Interpretation note: These values quantify average contribution magnitude, not direction or causality.
Model context: ROC-AUC=0.744, PR-AUC=0.388, F1@best_threshold=0.357.
