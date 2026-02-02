import pandas as pd
import numpy as np

df = pd.read_parquet("outputs/model_dataset.parquet")
winners = pd.read_csv("data/mvp_winners.csv")

# Clean whitespace
winners["player_name"] = winners["player_name"].astype(str).str.strip()
df["player_display_name"] = df["player_display_name"].astype(str).str.strip()

# Build abbreviated name like "T. Brady" from "Tom Brady"
def to_initial_last(full_name: str) -> str:
    parts = full_name.strip().split()
    if len(parts) < 2:
        return full_name.strip()
    first = parts[0]
    last = parts[-1]
    return f"{first[0]}. {last}"

df["name_initial_last"] = df["player_display_name"].apply(to_initial_last)

# Merge winners
df = df.merge(
    winners.rename(columns={"player_name": "winner_name"}),
    on="season",
    how="left"
)

# Target:
# - NaN if no winner (2025)
# - 1 if matches either full name OR abbreviated initials format
match_full = (df["player_display_name"] == df["winner_name"])
match_abbr = (df["name_initial_last"] == df["winner_name"])

df["is_mvp"] = np.where(
    df["winner_name"].isna(),
    np.nan,
    (match_full | match_abbr).astype(int)
)

df.to_parquet("outputs/model_dataset_labeled.parquet", index=False)

print("Saved outputs/model_dataset_labeled.parquet")
print("Seasons total:", df["season"].nunique())
print("Labeled seasons:", df[~df["is_mvp"].isna()]["season"].nunique())
print("Positive labels (should equal labeled seasons):", df["is_mvp"].sum())

# Debug a season
sample_season = winners["season"].iloc[0]
print("\nDEBUG — sample season check:")
print(
    df[df["season"] == sample_season][
        ["season", "player_display_name", "name_initial_last", "winner_name", "is_mvp"]
    ].head(15).to_string(index=False)
)

# Show seasons where MVP didn't match
bad = (
    df[~df["is_mvp"].isna()]
    .groupby("season")["is_mvp"]
    .sum()
    .reset_index()
    .query("is_mvp == 0")
)
if len(bad) > 0:
    print("\nSeasons with NO MVP match:")
    print(bad.to_string(index=False))
