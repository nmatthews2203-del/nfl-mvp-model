import pandas as pd

# Load wide SHAP values
shap = pd.read_csv("outputs/mvp_shap_values.csv")

# ID columns that identify a player-season
id_cols = ["season", "player_display_name", "recent_team"]

# SHAP value columns
value_cols = [c for c in shap.columns if c.startswith("shap_")]

# Convert to long format
shap_long = shap.melt(
    id_vars=id_cols,
    value_vars=value_cols,
    var_name="feature",
    value_name="shap_value"
)

# Clean feature names
shap_long["feature"] = shap_long["feature"].str.replace("shap_", "")

# Save
shap_long.to_csv("outputs/mvp_shap_long.csv", index=False)
print("Saved outputs/mvp_shap_long.csv")
