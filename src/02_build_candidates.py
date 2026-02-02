import pandas as pd

player = pd.read_parquet("outputs/player_season.parquet")

SEASON_COL = "season"
POS_COL = "position"
TEAM_COL = "recent_team"

PASS_ATT_COL = "attempts"
PASS_EPA_COL = "passing_epa"
RUSH_EPA_COL = "rushing_epa"
REC_EPA_COL = "receiving_epa"

RUSH_YDS_COL = "rushing_yards"
REC_YDS_COL = "receiving_yards"

# yards from scrimmage
player["yds_from_scrimmage"] = player[RUSH_YDS_COL].fillna(0) + player[REC_YDS_COL].fillna(0)

def build_candidates(df):
    out = []
    for season, sdf in df.groupby(SEASON_COL):

        # ---- QBs: include both high-value and high-volume ----
        qbs = sdf[sdf[POS_COL] == "QB"].copy()

        # Top by passing EPA
        qbs_epa = qbs.sort_values(PASS_EPA_COL, ascending=False).head(40)

        # Top by attempts (volume safety net)
        qbs_vol = qbs.sort_values(PASS_ATT_COL, ascending=False).head(40)

        qbs_final = pd.concat([qbs_epa, qbs_vol], axis=0).drop_duplicates(subset=["player_id"])

        # ---- Non-QBs: include impact + production ----
        non = sdf[sdf[POS_COL].isin(["RB", "WR", "TE"])].copy()

        non_prod = non.sort_values("yds_from_scrimmage", ascending=False).head(25)

        # Total EPA proxy (some tables may have missing rushing/receiving EPA)
        non["skill_epa"] = non[RUSH_EPA_COL].fillna(0) + non[REC_EPA_COL].fillna(0)
        non_impact = non.sort_values("skill_epa", ascending=False).head(25)

        non_final = pd.concat([non_prod, non_impact], axis=0).drop_duplicates(subset=["player_id"])

        out.append(pd.concat([qbs_final, non_final], axis=0))

    return pd.concat(out, axis=0)

candidates = build_candidates(player)

candidates.to_parquet("outputs/candidates.parquet", index=False)
print("Saved outputs/candidates.parquet")
print("Rows:", len(candidates))
print(candidates[["season", "player_display_name", "position", "recent_team"]].head(20).to_string(index=False))
