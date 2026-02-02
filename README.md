# NFL MVP Prediction Model (2010–2025)

End-of-season NFL MVP prediction model using **XGBoost**, with **SHAP explainability** and an interactive **Tableau** dashboard.

## What this project does
- Pulls and aggregates NFL player/team season data
- Labels historical MVP winners (2010–2024)
- Trains an MVP classifier (XGBoost)
- Produces MVP probabilities and SHAP explanations for analysis + Tableau

## Key results
- The model correctly identified the NFL MVP in each historical season from 2010–2024 when ranking candidates by predicted probability.
- For the 2025 season (no ground-truth MVP available), the model projects Drake Maye as the leading MVP candidate based on end-of-season performance metrics.
- MVP probabilities in competitive seasons are more evenly distributed, while runaway MVP seasons show dominant single-candidate probabilities.

## Model evaluation

### How the model was tested
The model was trained on historical NFL seasons from 2010–2024. Performance was evaluated by checking whether the player ranked highest by the model in each season matched the actual NFL MVP.

### How performance was measured
The primary focus was on how well the model ranked MVP candidates by probability, rather than making a single yes/no prediction. This reflects how MVP voting works in practice, where multiple players receive consideration.

### Baseline comparison
A simpler logistic regression model was used as a baseline. The final XGBoost model produced clearer separation between top candidates and more realistic probability distributions, particularly in competitive MVP races.

### What the results mean
In historical seasons, the top-ranked candidate aligned with the actual MVP. In seasons with a clear MVP favorite, the model assigned a high probability to one player, while in closer races the probabilities were more evenly distributed.

## Repository structure
- `src/` – Python pipeline scripts (numbered in run order)
- `data/` – Inputs (MVP winners CSV)
- `outputs/` – Generated datasets + Tableau-ready outputs
- `notebooks/` – Optional exploration (currently empty)

## Key outputs (Tableau-ready)
- `outputs/mvp_predictions_xgb.csv` – MVP probabilities by player-season (includes 2025)
- `outputs/mvp_shap_long.csv` – SHAP values (historical seasons only)

## Tableau dashboard
🔗 [View the interactive Tableau dashboard](https://public.tableau.com/app/profile/nicholas.matthews8591/viz/MVPPredictionModel/Dashboard1)

## Use of AI Tools
ChatGPT was used as a development assistant throughout this project to support debugging, code refactoring, feature engineering ideas, and visualization design decisions. All modeling choices, data interpretation, and final implementation decisions were made by the author.

## How to run
```bash
pip install -r requirements.txt

python src/01_pull_data.py
python src/02_build_candidates.py
python src/03_join_team_context.py
python src/04b_make_target.py
python src/06_train_xgboost.py
python src/07_pivot_shap.py

### What the dashboard shows
- MVP probability ladder (including 2025 prediction)
- SHAP-based feature explanations (historical seasons)
- Passing EPA vs team win percentage context
- MVP race competitiveness across seasons

