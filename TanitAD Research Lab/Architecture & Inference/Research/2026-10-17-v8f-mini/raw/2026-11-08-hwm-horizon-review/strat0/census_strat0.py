# STRAT0 census (Master Mind, 2026-11-08): what the published PhysicalAI-AV metadata offers for
# strategic context labels. Zero training, CPU, metadata only: no frame, no model.
# Layer: published corpus metadata (metadata/data_collection.parquet). Labels only: inference stays vision-only.
# Solar elevation is ESTIMATED: country-centroid latitude, mid-month declination, hour_of_day read as local solar
# time (no longitude or daylight-saving correction). Use it for class balance, never as a per-clip label.
import json, math, sys
import numpy as np, pandas as pd

ROOT = 'C:/Users/Admin/tanitad-data/physicalai/metadata/data_collection.parquet'
MC = 'E:/Projects/TanitAD-artifacts/dagland/motion_c'
SETS = {'train4512': MC + '/t2_scenes/raw/TRAIN_CLIPS_4512.json',
        'eval141': MC + '/theory/strat0/eval141_ids.json'}
OUT = MC + '/theory/strat0/CENSUS_STRAT0.json'
LAT = {'United States': 39, 'Germany': 51, 'France': 46.6, 'Italy': 42.8, 'Sweden': 62, 'Spain': 40, 'Portugal': 39.5,
       'Greece': 39, 'Austria': 47.5, 'Finland': 64, 'Croatia': 45, 'Netherlands': 52, 'Belgium': 50.5,
       'Switzerland': 46.8, 'Poland': 52, 'Czech Republic': 49.8, 'Czechia': 49.8, 'Denmark': 56, 'Norway': 61,
       'Hungary': 47, 'Romania': 46, 'Bulgaria': 42.7, 'Slovakia': 48.7, 'Slovenia': 46, 'Lithuania': 55,
       'Latvia': 57, 'Estonia': 59, 'Ireland': 53, 'United Kingdom': 54, 'Luxembourg': 49.8, 'Serbia': 44}
LEFT_HAND = {'United Kingdom', 'Ireland', 'Malta', 'Cyprus', 'Japan', 'Australia', 'New Zealand'}


def ids(path):
    d = json.load(open(path, encoding='utf8'))
    if isinstance(d, dict):
        lists = [v for v in d.values() if isinstance(v, list)]
        d = max(lists, key=len) if lists else list(d.keys())
    return [str(x['clip_id'] if isinstance(x, dict) else x) for x in d]


def cramers_v(a, b):
    t = pd.crosstab(a, b).to_numpy(float)
    n = t.sum()
    e = t.sum(0, keepdims=True) * t.sum(1, keepdims=True) / n
    chi2 = ((t - e) ** 2 / e).sum()
    return float(math.sqrt(chi2 / n / max(1, min(t.shape) - 1)))


def solar_el(lat, month, hour):
    doy = 30.4 * (month - 1) + 15
    dec = math.radians(-23.44 * math.cos(2 * math.pi * (doy + 10) / 365))
    phi = math.radians(lat)
    return math.degrees(math.asin(math.sin(phi) * math.sin(dec)
                                  + math.cos(phi) * math.cos(dec) * math.cos(math.radians(15 * (hour + 0.5 - 12)))))


def band(e):
    return 'night' if e < -6 else 'twilight' if e < 6 else 'low_sun' if e < 20 else 'day'


d = pd.read_parquet(ROOT)
d.index = d.index.astype(str)
res = {'n_all': int(len(d)), 'columns': list(d.columns), 'null_counts': {c: int(d[c].isna().sum()) for c in d.columns}}
res['countries'] = d['country'].value_counts().to_dict()
res['left_hand_traffic_present'] = {c: int(res['countries'].get(c, 0)) for c in LEFT_HAND if c in res['countries']}
res['unmapped_lat'] = [c for c in res['countries'] if c not in LAT]

# Is hour_of_day local? If it were one common clock, the daily trough would shift with longitude.
res['diurnal'] = {}
for c, n in d['country'].value_counts().items():
    if n < 3000:
        continue
    h = d.loc[d['country'] == c, 'hour_of_day'].value_counts().reindex(range(24), fill_value=0).to_numpy(float)
    w = np.array([h[[(i + j) % 24 for j in range(5)]].sum() for i in range(24)])
    res['diurnal'][str(c)] = {'n': int(n), 'trough_5h_start': int(w.argmin()), 'trough_share': round(float(w.min() / h.sum()), 4),
                              'peak_hour': int(h.argmax())}

res['cramers_v'] = {f'country_vs_{k}': round(cramers_v(d['country'], d[k]), 4) for k in ('platform_class', 'radar_config', 'month')}
res['platform_classes'] = {str(k): int(v) for k, v in d['platform_class'].value_counts().items()}
res['radar_configs_n'] = int(d['radar_config'].nunique())
top = d['country'].value_counts().head(6).index
res['platform_by_top_country'] = {str(c): {str(k): round(float(v), 3) for k, v in
                                           d.loc[d['country'] == c, 'platform_class'].value_counts(normalize=True).items()} for c in top}

m = d[d['country'].isin(LAT)].copy()
m['el'] = [solar_el(LAT[c], mo, h) for c, mo, h in zip(m['country'], m['month'], m['hour_of_day'])]
m['band'] = m['el'].map(band)
res['illum_all_ESTIMATED'] = {k: int(v) for k, v in m['band'].value_counts().items()}
res['n_mapped'] = int(len(m))
clock_day = m['hour_of_day'].between(7, 18)
res['clock_vs_sun_ESTIMATED'] = {'clock_7to18_but_sun_below_minus6': int(((m['band'] == 'night') & clock_day).sum()),
                                 'clock_outside_7to18_but_sun_above_6': int(((m['band'].isin(['low_sun', 'day'])) & ~clock_day).sum()),
                                 'n': int(len(m))}

res['sets'] = {}
for name, p in SETS.items():
    x = ids(p)
    s = d.reindex(x)
    ok = s['country'].notna()
    e = [band(solar_el(LAT[c], mo, h)) if c in LAT else 'unmapped'
         for c, mo, h in zip(s.loc[ok, 'country'], s.loc[ok, 'month'], s.loc[ok, 'hour_of_day'])]
    res['sets'][name] = {'n': len(x), 'n_joined': int(ok.sum()), 'n_countries': int(s['country'].nunique()),
                         'country': {str(k): int(v) for k, v in s['country'].value_counts().items()},
                         'illum_ESTIMATED': {k: int(v) for k, v in pd.Series(e).value_counts().items()}}
json.dump(res, open(OUT, 'w', encoding='utf8'), indent=1, ensure_ascii=False)
print(json.dumps(res, ensure_ascii=False)[:6500])
