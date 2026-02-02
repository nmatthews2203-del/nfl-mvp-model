import pandas as pd
import nflreadpy as nfl

SEASONS = list(range(2010, 2026))

# Load data (Polars DataFrames)
player_pl = nfl.load_player_stats(seasons=SEASONS, summary_level="reg")
team_pl = nfl.load_team_stats(seasons=SEASONS, summary_level="reg")

# Convert to Pandas
player = player_pl.to_pandas()
team = team_pl.to_pandas()

print("player rows:", len(player), "cols:", len(player.columns))
print("team rows:", len(team), "cols:", len(team.columns))

player.to_parquet("outputs/player_season.parquet", index=False)
team.to_parquet("outputs/team_season.parquet", index=False)

print("Saved outputs/*.parquet")
