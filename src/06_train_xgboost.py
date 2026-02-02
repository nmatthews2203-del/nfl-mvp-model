import pandas as pd
import numpy as np

import xgboost as xgb
import shap
import joblib

from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

# Load data
df = pd.read_parquet("outputs/model_dataset_labeled.parquet")

train = df[~df["is_mvp"].isna()].copy()
pred = df[df["is_mvp"].isna()].copy()

y = train["is_mvp"].astype(int)
groups = train["season"]

FEATURES = [
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "passing_epa",
    "passing_cpoe",
    "rushing_yards",
    "rushing_tds",
    "rushing_epa",
    "receiving_yards",
    "receiving_tds",
    "receiving_epa",
    "win_pct",
]

FEATURES = [f for f in FEATURES if f in train.columns]
X = train[FEATURES].fillna(0)

print("Using features:", FEATURES)

# XGBoost model
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
)

# Cross-validation by season
gkf = GroupKFold(n_splits=5)

oof = np.zeros(len(train))
losses = []

for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), 1):
    model.fit(X.iloc[tr], y.iloc[tr])
    oof[te] = model.predict_proba(X.iloc[te])[:, 1]
    loss = log_loss(y.iloc[te], oof[te], labels=[0, 1])
    losses.append(loss)
    print(f"Fold {fold} logloss: {loss:.4f}")

print("Avg logloss:", float(np.mean(losses)))

# Fit on all labeled seasons
model.fit(X, y)
joblib.dump(model, "outputs/mvp_xgboost.joblib")
print("Saved outputs/mvp_xgboost.joblib")

# Predictions for historical seasons
train["mvp_prob"] = model.predict_proba(X)[:, 1]
train["mvp_rank"] = train.groupby("season")["mvp_prob"].rank(ascending=False, method="first")

# Evaluation
top1 = train[train["mvp_rank"] == 1]
top1_acc = (top1["is_mvp"] == 1).mean()

top3 = train[train["mvp_rank"] <= 3].groupby("season")["is_mvp"].max().mean()

print("Top-1 accuracy:", round(float(top1_acc), 3))
print("Top-3 hit rate:", round(float(top3), 3))

# Predict future season(s)
if len(pred) > 0:
    pred["mvp_prob"] = model.predict_proba(pred[FEATURES].fillna(0))[:, 1]
    pred["mvp_rank"] = pred.groupby("season")["mvp_prob"].rank(ascending=False, method="first")

# Combine
final = pd.concat([train, pred], axis=0)

# ---- SHAP EXPLAINABILITY ----
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap_df = pd.DataFrame(
    shap_values,
    columns=[f"shap_{f}" for f in FEATURES]
)

shap_out = pd.concat(
    [train.reset_index(drop=True)[["season", "player_display_name", "recent_team"]],
     shap_df],
    axis=1
)

shap_out.to_csv("outputs/mvp_shap_values.csv", index=False)
print("Saved outputs/mvp_shap_values.csv")

# Export predictions for Tableau
keep_cols = [
    "season", "player_display_name", "position", "recent_team",
    "wins", "losses", "win_pct",
    "mvp_prob", "mvp_rank", "is_mvp",
] + FEATURES

final[keep_cols].to_csv("outputs/mvp_predictions_xgb.csv", index=False)
print("Saved outputs/mvp_predictions_xgb.csv")
