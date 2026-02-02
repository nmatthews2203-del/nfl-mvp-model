import pandas as pd

cand = pd.read_parquet("outputs/candidates.parquet")
team = pd.read_parquet("outputs/team_season.parquet")

print("TEAM COLUMNS (first 60):")
print(list(team.columns)[:60])
print("\nTEAM COLUMNS (contains 'win' or 'loss'):")
print([c for c in team.columns if "win" in c.lower() or "loss" in c.lower()])

# ---- Confirmed from your player table ----
SEASON_COL = "season"
TEAM_COL_CAND = "recent_team"

# ---- Find likely columns in team table ----
# Common possibilities in nflverse tables
possible_team_cols = ["team", "recent_team", "team_abbr", "team_id"]
possible_wins = ["wins", "win", "w", "team_wins", "total_wins", "n_wins"]
possible_losses = ["losses", "loss", "l", "team_losses", "total_losses", "n_losses"]

def first_existing(cols, options):
    for o in options:
        if o in cols:
            return o
    return None

cols = set(team.columns)

TEAM_COL_TEAM = first_existing(cols, possible_team_cols)
WINS_COL = first_existing(cols, possible_wins)
LOSSES_COL = first_existing(cols, possible_losses)

print("\nAuto-detected:")
print("TEAM_COL_TEAM =", TEAM_COL_TEAM)
print("WINS_COL      =", WINS_COL)
print("LOSSES_COL    =", LOSSES_COL)

if TEAM_COL_TEAM is None or WINS_COL is None or LOSSES_COL is None:
    raise ValueError(
        "Could not auto-detect team/wins/losses columns. "
        "Paste the printed column list here and I'll map it instantly."
    )

team_small = team[[SEASON_COL, TEAM_COL_TEAM, WINS_COL, LOSSES_COL]].copy()
team_small = team_small.rename(columns={TEAM_COL_TEAM: TEAM_COL_CAND})

df = cand.merge(team_small, on=[SEASON_COL, TEAM_COL_CAND], how="left")
df["win_pct"] = df[WINS_COL] / (df[WINS_COL] + df[LOSSES_COL])

df.to_parquet("outputs/model_dataset.parquet", index=False)

print("\nSaved outputs/model_dataset.parquet")
print(df[[SEASON_COL, TEAM_COL_CAND, WINS_COL, LOSSES_COL, "win_pct"]].head())
print("Missing win_pct rows:", df["win_pct"].isna().sum())