"""Score everything the closed-loop dump enables: run-6 gates, closed-loop verdicts,
and the decisive vision-only S-curve arm. Runs on the dev box from dump_cl npz."""
import glob, json, math
import numpy as np

DT, THR, NEAR = 0.1, 0.03, 3

def controls(path):
    p = np.concatenate([np.zeros((path.shape[0],1,2)), path[...,:2]], 1)
    d = p[:,1:]-p[:,:-1]
    ds = np.sqrt((d**2).sum(-1)+1e-12)
    sp = ds/DT
    acc = (sp[:,1:]-sp[:,:-1])/DT
    h = np.arctan2(d[...,1], d[...,0])
    dh = (h[:,1:]-h[:,:-1]+math.pi)%(2*math.pi)-math.pi
    ok = (ds[:,1:]>0.05)&(ds[:,:-1]>0.05)
    return sp, acc, np.where(ok, dh, 0.0).sum(1), dh, ok

def fam_row(P, G):
    sp_p, ac_p, ny_p, dh_p, ok_p = controls(P)
    sp_g, ac_g, ny_g, dh_g, ok_g = controls(G)
    jerk = (ac_p[:,1:]-ac_p[:,:-1])/DT
    return {"ade_m": float(np.linalg.norm(P[...,:2]-G[...,:2], axis=-1).mean()),
            "speed_bias_mps": float((sp_p-sp_g).mean()),
            "speed_mae_mps": float(np.abs(sp_p-sp_g).mean()),
            "accel_rms_mps2": float(np.sqrt((ac_p**2).mean())),
            "jerk_rms_mps3": float(np.sqrt((jerk**2).mean())),
            "net_yaw_err_rad": float(np.abs(ny_p-ny_g).mean())}

def boot(per_a, per_b, keys, n=2000):
    rng = np.random.default_rng(0); E=len(per_a); out={}
    for k in keys:
        d = np.array([e[k] for e in per_b]) - np.array([e[k] for e in per_a])
        dr = [float(d[rng.integers(0,E,E)].mean()) for _ in range(n)]
        lo,hi = np.percentile(dr,[2.5,97.5])
        out[k]={"delta":float(d.mean()),"lo":float(lo),"hi":float(hi),
                "separated":bool(lo>0 or hi<0)}
    return out

dumps = sorted(glob.glob("cl_dump/ep*.npz"))
assert len(dumps)==40, len(dumps)
old = sorted(glob.glob("v16dump/ep*.npz"))
ARMS = ("o16","o6","c16","c6","h16")
per = {a: [] for a in ARMS}
ev = {a: {"decel": [], "accel": []} for a in ARMS}
ev_gt = {"decel": [], "accel": []}
lag = {a: [] for a in ARMS}
s_hit = {a: [] for a in ARMS}; s_tot=0
match_b = []
for f, fo in zip(dumps, old):
    d = np.load(f); do = np.load(fo)
    G = d["g"].astype(np.float64)
    match_b.append(float(np.abs(d["o16"]-do["b"][..., :d["o16"].shape[-1]]).max()))
    g1o, g2o, gdh = None, None, None
    _, ag, gny, gdhh, _ = controls(G)
    a_g = ag[:, :NEAR].mean(1)
    m_de, m_ac = a_g < -1.0, a_g > 1.0
    ev_gt["decel"] += list(a_g[m_de]); ev_gt["accel"] += list(a_g[m_ac])
    # S definition on GT
    h1 = gdhh[:, :9].sum(1); h2 = gdhh[:, 9:].sum(1)
    is_s = (np.sign(h1)!=np.sign(h2)) & (np.abs(h1)>THR) & (np.abs(h2)>THR)
    s_tot += int(is_s.sum())
    for a in ARMS:
        P = d[a].astype(np.float64)
        per[a].append(fam_row(P, G))
        _, ap, _, pdh, _ = controls(P)
        a_p = ap[:, :NEAR].mean(1)
        ev[a]["decel"] += list(a_p[m_de]); ev[a]["accel"] += list(a_p[m_ac])
        x, y = a_p - a_p.mean(), a_g - a_g.mean()
        den = math.sqrt((x**2).sum()*(y**2).sum())
        if den > 1e-9:
            cs = [ (float((x[l:]*y[:len(y)-l]).sum()) if l>=0 else float((x[:l]*y[-l:]).sum()))/den
                   for l in range(-20,21)]
            if max(cs) > 0.2: lag[a].append((int(np.argmax(cs))-20)*DT)
        p1 = pdh[:, :9].sum(1); p2 = pdh[:, 9:].sum(1)
        hit = (np.sign(p1)==np.sign(h1)) & (np.sign(p2)==np.sign(h2)) & \
              (np.abs(p1)>THR/2) & (np.abs(p2)>THR/2)
        s_hit[a] += list(hit[is_s])

out = {"_o16_vs_banked_b_max_abs": max(match_b), "n_s_windows": s_tot}
gt_de = float(np.mean(ev_gt["decel"])); gt_ac = float(np.mean(ev_gt["accel"]))
for a in ARMS:
    P_all = {k: float(np.mean([r[k] for r in per[a]])) for k in per[a][0]}
    row = dict(P_all)
    row["decel_response_ratio"] = round(float(np.mean(ev[a]["decel"]))/gt_de, 4)
    row["accel_response_ratio"] = round(float(np.mean(ev[a]["accel"]))/gt_ac, 4)
    row["lag_accel_s_mean"] = round(float(np.mean(lag[a])), 4) if lag[a] else None
    row["s_reproduction_rate"] = round(float(np.mean(s_hit[a])), 4)
    out[a] = {k: round(v,4) if isinstance(v,float) else v for k,v in row.items()}
    print(a, json.dumps(out[a]))
out["paired_run6_minus_v16_open"] = boot(per["o16"], per["o6"],
    ("ade_m","speed_mae_mps","jerk_rms_mps3","net_yaw_err_rad"))
out["paired_closed_minus_open_v16"] = boot(per["o16"], per["c16"], ("ade_m","speed_mae_mps"))
out["paired_closed_minus_open_run6"] = boot(per["o6"], per["c6"], ("ade_m","speed_mae_mps"))
for k in ("paired_run6_minus_v16_open","paired_closed_minus_open_v16","paired_closed_minus_open_run6"):
    print(k, json.dumps(out[k]))
json.dump(out, open("closed_loop_analysis.json","w"), indent=1)
print("ANALYZE_DONE")
