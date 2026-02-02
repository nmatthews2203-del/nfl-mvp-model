import pandas as pd

cand = pd.read_parquet("outputs/candidates.parquet")
record = pd.read_parquet("outputs/team_record.parquet")

merged = cand.merge(
    record,
    left_on=["season", "recent_team"],
    right_on=["season", "team"],
    how="left",
    indicator=True
)

missing = merged[merged["_merge"] == "left_only"].copy()

print("Missing rows:", len(missing))
print("\nUnique (season, recent_team) missing combos:")
print(
    missing[["season", "recent_team"]]
    .drop_duplicates()
    .sort_values(["season", "recent_team"])
    .to_string(index=False)
)

print("\nCount by recent_team:")
print(missing["recent_team"].value_counts().to_string())
