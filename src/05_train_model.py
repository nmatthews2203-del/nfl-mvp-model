import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import joblib

# Load labeled dataset
df = pd.read_parquet("outputs/model_dataset_labeled.parquet")

# Split into:
# - train: seasons with known MVP (is_mvp not null)
# - predict: future season(s) like 2025 (is_mvp is null)
train = df[~df["is_mvp"].isna()].copy()
pred = df[df["is_mvp"].isna()].copy()

# Convert target to int for sklearn
y = train["is_mvp"].astype(int)

# Pick a strong but simple feature set
# (all of these exist in your columns list)
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

# Keep only features that actually exist (safety)
FEATURES = [f for f in FEATURES if f in train.columns]

print("Using features:", FEATURES)

X = train[FEATURES].fillna(0)
groups = train["season"]

# Model pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
])

# Cross-val by season (no leakage)
gkf = GroupKFold(n_splits=5)

oof = np.zeros(len(train))
losses = []

for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    oof[te_idx] = model.predict_proba(X.iloc[te_idx])[:, 1]
    fold_loss = log_loss(y.iloc[te_idx], oof[te_idx], labels=[0, 1])
    losses.append(fold_loss)
    print(f"Fold {fold} logloss: {fold_loss:.4f}")

print("Avg logloss:", float(np.mean(losses)))

# Fit final model on all labeled seasons
model.fit(X, y)
joblib.dump(model, "outputs/mvp_logreg.joblib")
print("Saved outputs/mvp_logreg.joblib")

# Add predictions for labeled seasons (for evaluation + Tableau history)
train["mvp_prob"] = model.predict_proba(train[FEATURES].fillna(0))[:, 1]
train["mvp_rank"] = train.groupby("season")["mvp_prob"].rank(ascending=False, method="first")

# Evaluate: Top-1 + Top-3 hit rate
top1 = train[train["mvp_rank"] == 1]
top1_acc = (top1["is_mvp"] == 1).mean()

top3 = train[train["mvp_rank"] <= 3].groupby("season")["is_mvp"].max().mean()

print("Top-1 accuracy:", round(float(top1_acc), 3))
print("Top-3 hit rate:", round(float(top3), 3))

# Predict future season(s) (e.g., 2025)
if len(pred) > 0:
    pred["mvp_prob"] = model.predict_proba(pred[FEATURES].fillna(0))[:, 1]
    pred["mvp_rank"] = pred.groupby("season")["mvp_prob"].rank(ascending=False, method="first")

# Combine and export for Tableau
final = pd.concat([train, pred], axis=0, ignore_index=True)

# Keep a clean set of columns for Tableau (you can expand later)
keep_cols = [
    "season", "player_id", "player_display_name", "position", "recent_team",
    "wins", "losses", "ties", "win_pct",
    "mvp_prob", "mvp_rank", "is_mvp",
    "passing_yards", "passing_tds", "passing_interceptions", "passing_epa", "passing_cpoe",
    "rushing_yards", "rushing_tds", "rushing_epa",
    "receiving_yards", "receiving_tds", "receiving_epa",
]
keep_cols = [c for c in keep_cols if c in final.columns]

final[keep_cols].to_csv("outputs/mvp_predictions.csv", index=False)
print("Saved outputs/mvp_predictions.csv")
