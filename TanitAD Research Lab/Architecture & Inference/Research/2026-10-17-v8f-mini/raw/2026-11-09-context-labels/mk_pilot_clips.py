"""Deterministic PILOT clip selection (seed string 'pilot-v1'), ~200 clips from train4512 + eval (video_local only).
Includes the 20 smoke clips and at least 10 eval clips; stratified over country x {night, day} (hour used ONLY as stratifier: night = hour >= 20 or < 6),
filled round-robin over the cells in stable-hash order. Does not change any existing selection list. Writes splits/pilot_clips.json + PILOT_SELECTION.json."""
import json, os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strat_common as C
N = 200
t = pd.read_parquet(HERE + "/splits/clips_table.parquet")
ids_all = set(json.load(open(HERE + "/splits/clips_all.json"))["clips"])
pool = t[t.split.isin(["train4512", "eval"]) & t.video_local & t.clip_id.isin(ids_all)].copy()
pool["band"] = ["night" if (h >= 20 or h < 6) else "day" for h in pool.hour_of_day]
pool["h"] = [C.stable_hash(c, "pilot-v1") for c in pool.clip_id]
pool = pool.sort_values("h")
smoke = json.load(open(HERE + "/smoke/smoke20.json"))["clips"]
sel = list(smoke)
ev = pool[(pool.split == "eval") & ~pool.clip_id.isin(sel)]
for _, g in ev.groupby("country"):
    if sum(pool.set_index("clip_id").loc[sel].split == "eval") >= 10:
        break
    sel.append(g.clip_id.iloc[0])
cells = {k: list(g.clip_id) for k, g in pool.groupby(["country", "band"])}
cur = {k: 0 for k in cells}
while len(sel) < N:
    moved = False
    for k in sorted(cells):
        v = cells[k]
        while cur[k] < len(v) and v[cur[k]] in sel:
            cur[k] += 1
        if cur[k] < len(v) and len(sel) < N:
            sel.append(v[cur[k]]); moved = True
    if not moved:
        break
sub = pool.set_index("clip_id").loc[sel]
rep = {"n": len(sel), "n_smoke": len(smoke), "split": sub.split.value_counts().to_dict(), "band": sub.band.value_counts().to_dict(),
       "country": sub.country.value_counts().to_dict(), "country_x_band": {f"{a}|{b}": int(n) for (a, b), n in sub.groupby(["country", "band"]).size().items()},
       "pool_n": len(pool), "seed": "pilot-v1"}
json.dump({"clips": sel}, open(HERE + "/splits/pilot_clips.json", "w"))
json.dump(rep, open(HERE + "/splits/PILOT_SELECTION.json", "w"), indent=1)
print(json.dumps({k: rep[k] for k in ("n", "n_smoke", "split", "band", "pool_n")}), len(rep["country"]), "countries; min/max per country", min(rep["country"].values()), max(rep["country"].values()))
