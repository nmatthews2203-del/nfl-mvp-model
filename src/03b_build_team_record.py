import pandas as pd
import numpy as np

sch = pd.read_parquet("outputs/schedules.parquet")

# Keep regular season games only (column names vary slightly; handle both)
if "game_type" in sch.columns:
    sch = sch[sch["game_type"] == "REG"]
elif "season_type" in sch.columns:
    sch = sch[sch["season_type"].isin(["REG", "Regular", "regular"])]

# We only want games that have final scores
# Common columns: home_score, away_score
home_score_col = "home_score" if "home_score" in sch.columns else None
away_score_col = "away_score" if "away_score" in sch.columns else None

if home_score_col is None or away_score_col is None:
    raise ValueError("Could not find home_score/away_score columns. Paste the schedule columns sample.")

sch = sch.dropna(subset=[home_score_col, away_score_col]).copy()

# Common team columns: home_team, away_team
home_team_col = "home_team" if "home_team" in sch.columns else None
away_team_col = "away_team" if "away_team" in sch.columns else None

if home_team_col is None or away_team_col is None:
    raise ValueError("Could not find home_team/away_team columns. Paste the schedule columns sample.")

# Determine winners
sch["home_win"] = (sch[home_score_col] > sch[away_score_col]).astype(int)
sch["away_win"] = (sch[away_score_col] > sch[home_score_col]).astype(int)
sch["tie"] = (sch[home_score_col] == sch[away_score_col]).astype(int)

# Build team-game rows for aggregation
home_rows = sch[["season", home_team_col, "home_win", "away_win", "tie"]].copy()
home_rows = home_rows.rename(columns={home_team_col: "team"})
home_rows["wins"] = home_rows["home_win"]
home_rows["losses"] = home_rows["away_win"]

away_rows = sch[["season", away_team_col, "home_win", "away_win", "tie"]].copy()
away_rows = away_rows.rename(columns={away_team_col: "team"})
away_rows["wins"] = away_rows["away_win"]
away_rows["losses"] = away_rows["home_win"]

team_games = pd.concat([home_rows[["season", "team", "wins", "losses", "tie"]],
                        away_rows[["season", "team", "wins", "losses", "tie"]]],
                       axis=0)

team_record = team_games.groupby(["season", "team"], as_index=False).agg(
    wins=("wins", "sum"),
    losses=("losses", "sum"),
    ties=("tie", "sum"),
    games=("wins", "size"),
)

team_record["win_pct"] = (team_record["wins"] + 0.5 * team_record["ties"]) / (
    team_record["wins"] + team_record["losses"] + team_record["ties"]
)

team_record.to_parquet("outputs/team_record.parquet", index=False)
print("Saved outputs/team_record.parquet")
print(team_record.head())