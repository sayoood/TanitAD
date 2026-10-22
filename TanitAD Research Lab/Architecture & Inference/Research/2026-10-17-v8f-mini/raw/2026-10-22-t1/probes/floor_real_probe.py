"""REAL-DATA, CAMERA-FREE probe of the T1-kin loop: the P10 FLOOR (the checkpoint's frozen prior) driven by its own motion on the 141 real
eval clips (native 10 Hz tracks from eval141_kin.npz). The floor needs no image feature, so this runs without the pooled eval cache.
Reports: (a) m=30 == the open-loop floor ADE (a pipeline check against the banked P10 number), (b) the T1 floor at m = 1, 5, 10,
(c) the fidelity of the Hermite state rebuild: oracle plan = the logged future knots -> compare the rebuilt (v0, a0, kappa0) with the
LOGGED ones at every tick."""
import sys, json, time
import numpy as np, torch
import v8f_t1_eval as T1
from tanitad.data import v8f_mini_data as D
from tanitad.models import v8f_mini as M
from taniteval import floors_kin as fk

raw, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
z = np.load(raw + "/eval141_kin.npz"); grid = json.load(open(raw + "/eval_grid_windows.json"))
off = np.concatenate([[0], np.cumsum(z["n"])])
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
prior = M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(60)); 
prior.load_state_dict({k[len("prior."):]: v for k, v in ck["model"].items() if k.startswith("prior.")})

def base_dict(track):
    pg = track[0::2].copy(); T = len(pg)
    kin3, om = D.yaw_kin_from_track(track)
    p10 = np.full((T, 8), np.nan, np.float32); p10[1:] = fk.kin_features(kin3[1:]).astype(np.float32)
    d = {"pose": pg, "kin3": kin3, "p10f": p10, "a": np.zeros(T, np.float32), "ch1": np.nan_to_num(om).astype(np.float32),
         "v": pg[:, 3].astype(np.float32), "base": D.kdx_base(kin3, "kappa")}
    d["gt"], d["gt_ok"] = D.gt_future(pg)
    return d

tracks, bases, wins = {}, {}, []
for i, c in enumerate(grid["order"]):
    tr = z["poses"][off[i]:off[i + 1]].astype(np.float64); tracks[i] = tr; bases[i] = base_dict(tr)
    wins += [(i, int(t)) for t in grid["windows"][c]]
print("clips", len(tracks), "windows", len(wins))
res = {}
def ade(exe, gts): return float(np.linalg.norm(exe - gts, axis=-1).mean())
gts = np.stack([bases[i]["gt"][t] for i, t in wins])

import os
for m in ((30, 10, 5, 2, 1) if not os.environ.get('SKIP_SWEEP') else ()):
    t_start = time.time()
    states = [T1.WindowState(bases[i], tracks[i], np.zeros((len(bases[i]["pose"]), 5), np.float32), i, t, "logged") for i, t in wins]
    for n in range(30 // m):
        pf = torch.as_tensor(np.stack([s.d["p10f"][s.j] for s in states]), dtype=torch.float32)
        bs = torch.as_tensor(np.stack([s.d["base"][s.j] for s in states]), dtype=torch.float32)
        with torch.no_grad(): plan = prior(pf, bs).numpy()
        for s, p in zip(states, plan): s.commit_plan(p.astype(np.float64), m)
    exe = np.stack([s.executed_ego() for s in states])
    d = np.linalg.norm(exe - gts, axis=-1)
    res[f"m{m}"] = {"ade6": float(d.mean()), "ade2": float(d[:, :10].mean()), "fde6": float(d[:, -1].mean()),
                    "windows_envelope_clipped": int(sum(s.clipped > 0 for s in states)), "median_ade6": float(np.median(d.mean(1))),
                    "p90_ade6": float(np.percentile(d.mean(1), 90)), "secs": round(time.time() - t_start, 1)}
    print(f"m={m}", res[f"m{m}"], flush=True)

# (c) fidelity of the rebuilt kinematics: oracle plan = logged future knots, m=1, 15 ticks, windows with room for 45 + 30 steps
sub = [(i, t) for i, t in wins if t + 45 < len(bases[i]["pose"])][::5]
states = [T1.WindowState(bases[i], tracks[i], np.zeros((len(bases[i]["pose"]), 5), np.float32), i, t, "logged") for i, t in sub]
for n in range(15):
    for s_ in states:
        p = fk.gt_waypoints(bases[s_.ci]["pose"], np.array([s_.j]), np.arange(1, 31))[0]
        s_.commit_plan(p, 1)
errs = {"v0": [], "a0": [], "kappa0": []}; lgs = {"v0": [], "a0": [], "kappa0": []}
for s_ in states:
    for j in range(s_.t0 + 1, s_.t0 + 15):
        lg = bases[s_.ci]["kin3"][j]; rb = s_.d["kin3"][j]
        for q, k in enumerate(("v0", "a0", "kappa0")):
            errs[k].append(rb[q] - lg[q]); lgs[k].append(lg[q])
fid = {}
for k in errs:
    v, l = np.array(errs[k]), np.array(lgs[k])
    fid[k] = {"rebuilt_minus_logged_median_abs": float(np.median(np.abs(v))), "p90_abs": float(np.percentile(np.abs(v), 90)),
              "logged_std": float(np.std(l)), "logged_median_abs": float(np.median(np.abs(l))), "corr": float(np.corrcoef(v + l, l)[0, 1]), "n": int(len(v))}
res["rebuild_fidelity_oracle_plan_m1"] = fid
print(json.dumps(fid, indent=1))
json.dump(res, open(out, "w"), indent=1)
