import pandas as pd
import nflreadpy as nfl

SEASONS = list(range(2010, 2026))

sch_pl = nfl.load_schedules(seasons=SEASONS)
sch = sch_pl.to_pandas()

sch.to_parquet("outputs/schedules.parquet", index=False)
print("Saved outputs/schedules.parquet")
print("rows:", len(sch), "cols:", len(sch.columns))
print("sample cols:", list(sch.columns)[:35])