"""GEO10K step 5: metadata-only context fields for the 10,000 clips, the way merge_context.py builds them:
country / month / hour_of_day = NVIDIA metadata/data_collection.parquet; period_of_day = merge_context.period_of_day(hour);
sun_band_est = merge_context.sun_band(sun_el_deg(country, month, hour)). Same 18 columns and dtypes as
tanitad-context-labels-v1 clips/clips.parquet. Image-derived columns are null (they need frames + a GPU pass).
CONTROL: the same code on EVERY clip of v1 (4,659) must reproduce v1's metadata columns."""
import glob, json, math, os
import numpy as np
import pandas as pd

OUT = "/root/geo10k/out"
LAT = {'Austria': 47.5, 'Belgium': 50.6, 'Bulgaria': 42.7, 'Croatia': 45.1, 'Czechia': 49.8, 'Denmark': 56.0, 'Estonia': 58.7, 'Finland': 64.0,
       'France': 46.6, 'Germany': 51.1, 'Greece': 39.0, 'Hungary': 47.2, 'Italy': 42.8, 'Latvia': 56.9, 'Lithuania': 55.3, 'Luxembourg': 49.8,
       'Netherlands': 52.2, 'Poland': 52.1, 'Portugal': 39.6, 'Romania': 45.9, 'Slovakia': 48.7, 'Slovenia': 46.1, 'Spain': 40.2, 'Sweden': 62.0,
       'United States': 39.8}
CUM = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)


def sun_el_deg(country, month, hour):
    doy = CUM[month - 1] + 15
    dec = math.radians(23.44) * math.sin(2 * math.pi * (doy - 81) / 365.0)
    lat = math.radians(LAT[country])
    ha = math.radians(15.0 * (hour + 0.5 - 12.0))
    return math.degrees(math.asin(math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha)))


def sun_band(el):
    return "night" if el < -6 else "twilight" if el < 0 else "low_sun" if el < 15 else "day"


def period_of_day(h):
    return "night" if (h >= 21 or h < 5) else "morning" if h < 11 else "midday" if h < 15 else "afternoon" if h < 18 else "evening"


def rows_for(clips, meta, chunk_of):
    out = []
    n_nocountry = 0
    for c in clips:
        m = meta.loc[c]
        country, month, hour = str(m["country"]), int(m["month"]), int(m["hour_of_day"])
        try:
            sb = sun_band(sun_el_deg(country, month, hour))
        except KeyError:
            sb = None
            n_nocountry += 1
        out.append({"clip_id": c, "country": country, "month": month, "hour_of_day": hour, "period_of_day": period_of_day(hour),
                    "sun_band_est": sb, "chunk": int(chunk_of[c])})
    return pd.DataFrame(out), n_nocountry


meta = pd.read_parquet("/root/datascale/meta/metadata/data_collection.parquet")
ci = pd.read_parquet("/root/datascale/meta/clip_index.parquet")
v1 = pd.read_parquet("/root/geo10k/ref/tanitad-context-labels-v1/clips/clips.parquet")

# ---- control on every v1 clip
mine, _ = rows_for(list(v1.clip_id), meta, ci["chunk"].to_dict())
ctl = {}
for col in ("country", "month", "hour_of_day", "period_of_day", "sun_band_est", "chunk"):
    a, b = v1[col].reset_index(drop=True), mine[col].reset_index(drop=True)
    eq = (a == b) | (a.isna() & b.isna())
    ctl[col] = {"n": int(len(a)), "n_equal": int(eq.sum())}
ctl["split_values_v1"] = v1.split.value_counts().to_dict()
ctl["version_values_v1"] = v1.label_version.value_counts().to_dict()
ctl["video_local_v1"] = v1.video_local.value_counts().to_dict()
json.dump(ctl, open(f"{OUT}/control_context.json", "w"), indent=1)
print("[g5] control vs v1:", json.dumps({k: v for k, v in ctl.items() if isinstance(v, dict) and "n" in v}), flush=True)
# ---- regression arm: shift one clip's hour by +12 and the control must go RED
bad = mine.copy()
bad.loc[0, "hour_of_day"] = (int(bad.loc[0, "hour_of_day"]) + 12) % 24
bad.loc[0, "period_of_day"] = period_of_day(int(bad.loc[0, "hour_of_day"]))
red = int((bad["hour_of_day"].reset_index(drop=True) != v1["hour_of_day"].reset_index(drop=True)).sum())
json.dump({"mutation": "clip 0 hour +12", "n_mismatch_hour": red, "went_red": red > 0}, open(f"{OUT}/regression_context.json", "w"))
print("[g5] regression arm (hour +12) mismatches:", red, flush=True)

# ---- the 10,000
sel = [l.strip() for l in open("/root/datascale/selection/clips_10k.txt") if l.strip()]
chunk_of = {}
for f in glob.glob("/root/datascale/keep10k/*/ledger_stage.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r.get("status") == "OK":
            chunk_of[r["clip"]] = int(r["chunk"])
assert all(chunk_of[c] == int(ci.loc[c, "chunk"]) for c in sel), "ledger chunk != clip_index chunk"
new, nnc = rows_for(sel, meta, chunk_of)
df = pd.DataFrame({
    "clip_id": new.clip_id, "vlm_clip": pd.Series([False] * len(new), dtype=v1.vlm_clip.dtype), "country": new.country,
    "weather_mode": pd.Series([None] * len(new), dtype=v1.weather_mode.dtype), "surface_mode": pd.Series([None] * len(new), dtype=v1.surface_mode.dtype),
    "visibility_mode": pd.Series([None] * len(new), dtype=v1.visibility_mode.dtype), "illumination_mode": pd.Series([None] * len(new), dtype=v1.illumination_mode.dtype),
    "period_of_day": new.period_of_day, "sun_band_est": new.sun_band_est,
    "illum_sun_flag_frac": pd.Series([np.nan] * len(new), dtype="float64"), "split": "datascale10k", "chunk": new.chunk.astype("int64"),
    "session_key": "NA", "month": new.month.astype("int64"), "hour_of_day": new.hour_of_day.astype("int64"),
    "video_local": pd.Series([False] * len(new), dtype=v1.video_local.dtype), "ego_local": pd.Series([False] * len(new), dtype=v1.ego_local.dtype),
    "label_version": "geo10k-meta-v1"})
assert list(df.columns) == list(v1.columns), (list(df.columns), list(v1.columns))
dt_ok = {c: str(df[c].dtype) == str(v1[c].dtype) for c in df.columns}
df.to_parquet(f"{OUT}/clips_datascale10k.parquet", index=False)
rep = {"n": len(df), "countries": int(df.country.nunique()), "n_country_not_in_LAT_table": nnc, "dtype_equal_to_v1": dt_ok,
       "period_counts": df.period_of_day.value_counts().to_dict(), "month_range": [int(df.month.min()), int(df.month.max())],
       "hour_range": [int(df.hour_of_day.min()), int(df.hour_of_day.max())], "null_columns": [c for c in df.columns if df[c].isna().all()]}
json.dump(rep, open(f"{OUT}/context_report.json", "w"), indent=1)
print("[g5]", json.dumps(rep)[:900])
print("ZZ GEO10K-G5 DONE ZZ")
