"""Is the T1 floor blow-up the PRIOR's feedback gain, or the noise of the rebuilt state? (real eval tracks, real P10 prior of M2b_s0.)
Oracle run: execute the LOGGED future knots (m=1); at every tick compute the floor's one-shot prediction twice -- from the LOGGED features
and from the REBUILT ones -- and score both against the GT of that tick. If the two ADE6 agree, rebuild noise is harmless to the map."""
import sys, json
import numpy as np, torch
import v8f_t1_eval as T1
from tanitad.data import v8f_mini_data as D
from tanitad.models import v8f_mini as M
from taniteval import floors_kin as fk
raw, ckpt, out = sys.argv[1:4]
z = np.load(raw + "/eval141_kin.npz"); grid = json.load(open(raw + "/eval_grid_windows.json")); off = np.concatenate([[0], np.cumsum(z["n"])])
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
prior = M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(60))
prior.load_state_dict({k[len("prior."):]: v for k, v in ck["model"].items() if k.startswith("prior.")})
def base_dict(track):
    pg = track[0::2].copy(); T = len(pg); kin3, om = D.yaw_kin_from_track(track)
    p10 = np.full((T, 8), np.nan, np.float32); p10[1:] = fk.kin_features(kin3[1:]).astype(np.float32)
    d = {"pose": pg, "kin3": kin3, "p10f": p10, "a": np.zeros(T, np.float32), "ch1": np.nan_to_num(om).astype(np.float32), "v": pg[:, 3].astype(np.float32), "base": D.kdx_base(kin3, "kappa")}
    d["gt"], d["gt_ok"] = D.gt_future(pg); return d
tracks, bases, wins = {}, {}, []
for i, c in enumerate(grid["order"]):
    tr = z["poses"][off[i]:off[i + 1]].astype(np.float64); tracks[i] = tr; bases[i] = base_dict(tr); wins += [(i, int(t)) for t in grid["windows"][c]]
sub = [(i, t) for i, t in wins if t + 45 < len(bases[i]["pose"])][::5]
states = [T1.WindowState(bases[i], tracks[i], np.zeros((len(bases[i]["pose"]), 5), np.float32), i, t, "logged") for i, t in sub]
ad_log, ad_reb = [], []
for n in range(15):
    for s in states:
        p = fk.gt_waypoints(bases[s.ci]["pose"], np.array([s.j]), np.arange(1, 31))[0]
        s.commit_plan(p, 1)
    for s in states:
        j = s.j
        gt = bases[s.ci]["gt"][j]
        with torch.no_grad():
            pl = prior(torch.as_tensor(bases[s.ci]["p10f"][j][None]), torch.as_tensor(bases[s.ci]["base"][j][None])).numpy()[0]
            pr = prior(torch.as_tensor(s.d["p10f"][j][None]), torch.as_tensor(s.d["base"][j][None])).numpy()[0]
        ad_log.append(np.linalg.norm(pl - gt, axis=-1).mean()); ad_reb.append(np.linalg.norm(pr - gt, axis=-1).mean())
r = {"n": len(ad_log), "ade6_floor_from_logged_state": float(np.mean(ad_log)), "ade6_floor_from_rebuilt_state": float(np.mean(ad_reb)),
     "ratio": float(np.mean(ad_reb) / np.mean(ad_log))}
print(json.dumps(r, indent=1)); json.dump(r, open(out, "w"), indent=1)
