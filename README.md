# NFL MVP Prediction Model (2010–2025)

End-of-season NFL MVP prediction model using **XGBoost**, with **SHAP explainability** and an interactive **Tableau** dashboard.

## What this project does
- Pulls and aggregates NFL player/team season data
- Labels historical MVP winners (2010–2024)
- Trains an MVP classifier (XGBoost)
- Produces MVP probabilities and SHAP explanations for analysis + Tableau

## Repository structure
- `src/` – Python pipeline scripts (numbered in run order)
- `data/` – Inputs (MVP winners CSV)
- `outputs/` – Generated datasets + Tableau-ready outputs
- `notebooks/` – Optional exploration (currently empty)

## Key outputs (Tableau-ready)
- `outputs/mvp_predictions_xgb.csv` – MVP probabilities by player-season (includes 2025)
- `outputs/mvp_shap_long.csv` – SHAP values (historical seasons only)

## How to run
```bash
pip install -r requirements.txt

python src/01_pull_data.py
python src/02_build_candidates.py
python src/03_join_team_context.py
python src/04b_make_target.py
python src/06_train_xgboost.py
python src/07_pivot_shap.py

## Tableau dashboard
🔗 [View the interactive Tableau dashboard](https://public.tableau.com/app/profile/nicholas.matthews8591/viz/MVPPredictionModel/Dashboard1)

### What the dashboard shows
- MVP probability ladder (including 2025 prediction)
- SHAP-based feature explanations (historical seasons)
- Passing EPA vs team win percentage context
- MVP race competitiveness across seasons

