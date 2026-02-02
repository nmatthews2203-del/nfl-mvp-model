import pandas as pd

cand = pd.read_parquet("outputs/candidates.parquet")
record = pd.read_parquet("outputs/team_record.parquet")

# --- Historical team standardization by season ---
def normalize_team(row):
    team = row["recent_team"]
    season = row["season"]

    # Rams
    if team == "LA" and season < 2016:
        return "STL"

    # Chargers
    if team == "LAC" and season < 2017:
        return "SD"

    # Raiders
    if team == "LV" and season < 2020:
        return "OAK"

    return team

cand["team_std"] = cand.apply(normalize_team, axis=1)

# record already uses historical team codes correctly
record["team_std"] = record["team"]

df = cand.merge(
    record[["season", "team_std", "wins", "losses", "ties", "win_pct"]],
    on=["season", "team_std"],
    how="left"
)

df.to_parquet("outputs/model_dataset.parquet", index=False)

print("Saved outputs/model_dataset.parquet")
print("Missing win_pct rows:", df["win_pct"].isna().sum())

# sanity check
print(
    df[df["win_pct"].isna()][["season", "recent_team", "team_std"]]
    .drop_duplicates()
    .sort_values(["season", "recent_team"])
    .to_string(index=False)
)