import pandas as pd

df = pd.read_parquet("outputs/model_dataset.parquet")

# Keep just what we need
out = (
    df[["season", "player_name", "position", "recent_team"]]
    .drop_duplicates()
    .sort_values(["season", "player_name"])
)

out.to_csv("outputs/candidates_by_season.csv", index=False)
print("Saved outputs/candidates_by_season.csv")

# Also print a quick peek for one season
print(out[out["season"] == out["season"].min()].head(25).to_string(index=False))
