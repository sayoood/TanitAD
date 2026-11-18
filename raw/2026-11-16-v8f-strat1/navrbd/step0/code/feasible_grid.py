"""NAVRBD step 0: filter the 7,594-window RB grid to the windows the T1 tool can roll (gt_ok at t0: a complete 6 s future).  Same ClipStore the tool builds; writes eval_grid_x_feasible.json + counts."""
import sys, os, json, hashlib
import numpy as np, torch
T = "/root/strat1/navrb2/tree"; sys.path.insert(0, T + "/stack")
from tanitad.data import v8f_mini_data as D
I = "/root/strat1/navrbd_step0/out/inputs"
ck = torch.load("/root/strat1/navrbd_step0/ckpt/N1R_s0/ckpt_last.pt", map_location="cpu", weights_only=False)
TR = D.load_native_tracks("/root/dscale10k/in/labels/native_tracks_datascale10k.pt")
store = D.ClipStore("/root/strat1/navrbd_step0/cache_view", "eval", lru=8, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=TR)
g = json.load(open(I + "/eval_grid_x.json"))
keep, drop, nodrop_clip = {}, 0, 0
why = {"t_beyond_len": 0, "gt_not_ok": 0}
for c in g["order"]:
    d = store.get(c); ok = np.asarray(d["gt_ok"]); T_ = len(ok)
    ts = []
    for t in g["windows"][c]:
        if t >= T_: why["t_beyond_len"] += 1; drop += 1
        elif not bool(ok[t]): why["gt_not_ok"] += 1; drop += 1
        else: ts.append(t)
    if ts: keep[c] = ts
order = sorted(keep)
out = {"order": order, "windows": {c: keep[c] for c in order}, "n_windows": int(sum(len(v) for v in keep.values())), "source": g["source"] + "; restricted to windows with gt_ok at t0 (a complete 6 s future), the T1 tool's own precondition"}
p = I + "/eval_grid_x_feasible.json"; json.dump(out, open(p, "w"))
res = {"grid_windows": g["n_windows"], "grid_episodes": len(g["order"]), "feasible_windows": out["n_windows"], "feasible_episodes": len(order), "dropped": drop, "why": why,
       "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
json.dump(res, open(I + "/FEASIBLE_GRID.json", "w"), indent=1); print(json.dumps(res, indent=1))
