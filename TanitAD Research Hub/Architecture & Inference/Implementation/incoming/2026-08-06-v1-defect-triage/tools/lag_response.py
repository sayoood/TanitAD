"""Sayed's observation: v1.6 LAGS in deceleration/acceleration and sometimes curvature.

Locate and quantify it from the banked stride-1 dump (6,834 windows, no re-inference).
Per frame t, each arm's plan gives a near-term commanded profile; comparing that time
series against GT's over the episode measures (a) LAG — cross-correlation peak offset,
(b) UNDERSHOOT — response magnitude during GT events, (c) WITHIN-PLAN DELAY — where in
the 2 s horizon the arm's deceleration actually happens when GT decelerates NOW.

Signals per frame (origin frame, first 3 steps of the plan):
  a_near(t)  = mean accel over plan steps 1..3   (needs 4 speeds -> 5 points incl. origin)
  yr_near(t) = mean yaw-rate over plan steps 1..3
GT twin from the g rows of the same windows. Events: GT decel a<-1, accel a>+1,
curvature |yr|>0.1 rad/s (thresholds stated in output).
"""
import glob
import json

import numpy as np

DT = 0.1
NEAR = 3


def profiles(P):
    """P [n,20,>=2] -> speeds [n,20], accel [n,19], yawrate [n,19] along the plan."""
    p = np.concatenate([np.zeros((P.shape[0], 1, 2)), P[..., :2]], 1)
    d = p[:, 1:] - p[:, :-1]
    sp = np.linalg.norm(d, axis=-1) / DT
    acc = (sp[:, 1:] - sp[:, :-1]) / DT
    h = np.arctan2(d[..., 1], d[..., 0])
    dh = (h[:, 1:] - h[:, :-1] + np.pi) % (2 * np.pi) - np.pi
    ok = (sp[:, 1:] * DT > 0.05) & (sp[:, :-1] * DT > 0.05)
    yr = np.where(ok, dh / DT, 0.0)
    return sp, acc, yr


def xcorr_lag(x, y, max_lag=20):
    """Lag (frames) at which x best matches y shifted; +lag means x TRAILS y."""
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    if denom < 1e-9:
        return np.nan, np.nan
    best, bl = -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = float((x[lag:] * y[:len(y) - lag]).sum()) / denom
        else:
            c = float((x[:lag] * y[-lag:]).sum()) / denom
        if c > best:
            best, bl = c, lag
    return bl, best


dumps = sorted(glob.glob("/workspace/v16_eval/dump/ep*.npz"))
assert len(dumps) == 40
arms = ("v1arch", "v16")
lags = {a: {"accel": [], "yaw": []} for a in arms}
ev = {a: {k: [] for k in ("decel", "accel", "curv")} for a in arms}
ev_gt = {k: [] for k in ("decel", "accel", "curv")}
horizon_decel = {a: [] for a in arms}      # arm's accel profile in GT-decel-onset windows
horizon_gt = []
for f in dumps:
    d = np.load(f)
    G = d["g"].astype(np.float64)
    _, ag, yg = profiles(G)
    a_g = ag[:, :NEAR].mean(1)
    yr_g = yg[:, :NEAR].mean(1)
    for key, arr in (("v1arch", d["a"]), ("v16", d["b"])):
        P = arr.astype(np.float64)
        _, ap, yp = profiles(P)
        a_p = ap[:, :NEAR].mean(1)
        yr_p = yp[:, :NEAR].mean(1)
        la, ca = xcorr_lag(a_p, a_g)
        ly, cy = xcorr_lag(yr_p, yr_g)
        if np.isfinite(la) and ca > 0.2:
            lags[key]["accel"].append(la)
        if np.isfinite(ly) and cy > 0.2:
            lags[key]["yaw"].append(ly)
        m = a_g < -1.0
        ev[key]["decel"] += list(a_p[m])
        m2 = a_g > 1.0
        ev[key]["accel"] += list(a_p[m2])
        m3 = np.abs(yr_g) > 0.1
        ev[key]["curv"] += list(np.sign(yr_g[m3]) * yr_p[m3])
        onset = m & np.concatenate([[False], ~m[:-1]])
        horizon_decel[key].append(ap[onset])
    ev_gt["decel"] += list(a_g[a_g < -1.0])
    ev_gt["accel"] += list(a_g[a_g > 1.0])
    m3 = np.abs(yr_g) > 0.1
    ev_gt["curv"] += list(np.abs(yr_g[m3]))
    onset = (a_g < -1.0) & np.concatenate([[False], ~(a_g[:-1] < -1.0)])
    horizon_gt.append(ag[onset])

out = {"_signals": f"near-term = mean over plan steps 1..{NEAR} (0.1-0.3 s); "
                   "events: GT decel a<-1, accel a>+1, curvature |yr|>0.1 rad/s; "
                   "lag from per-episode cross-correlation (corr>0.2 admitted), "
                   "+lag = arm trails GT",
       "n_events": {k: len(v) for k, v in ev_gt.items()}}
rng = np.random.default_rng(0)
for a in arms:
    la = np.array(lags[a]["accel"])
    ly = np.array(lags[a]["yaw"])
    row = {"lag_accel_frames_mean": float(la.mean()) if la.size else None,
           "lag_accel_s_mean": float(la.mean() * DT) if la.size else None,
           "lag_accel_n_eps": int(la.size),
           "lag_yaw_frames_mean": float(ly.mean()) if ly.size else None,
           "lag_yaw_s_mean": float(ly.mean() * DT) if ly.size else None,
           "lag_yaw_n_eps": int(ly.size)}
    for k in ("decel", "accel", "curv"):
        g = np.array(ev_gt[k]) if k != "curv" else np.array(ev_gt[k])
        p = np.array(ev[a][k])
        gm = float(np.mean(np.abs(g))) if g.size else np.nan
        pm = float(np.mean(p * (np.sign(np.mean(g)) if k != "curv" else 1.0))) if p.size else np.nan
        row[f"{k}_gt_mean"] = round(float(np.mean(g)), 4) if g.size else None
        row[f"{k}_arm_mean"] = round(float(np.mean(p)), 4) if p.size else None
        row[f"{k}_response_ratio"] = (round(float(np.mean(p) / np.mean(g)), 4)
                                      if g.size and abs(np.mean(g)) > 1e-6 else None)
    H = np.concatenate([h for h in horizon_decel[a] if h.size], 0)
    row["decel_onset_horizon_accel_mean_by_step"] = [round(float(x), 3)
                                                     for x in H.mean(0)] if H.size else None
    out[a] = row
    print(f"[{a}] {json.dumps(row)[:400]}", flush=True)
Hg = np.concatenate([h for h in horizon_gt if h.size], 0)
out["gt_decel_onset_horizon_accel_mean_by_step"] = [round(float(x), 3)
                                                    for x in Hg.mean(0)] if Hg.size else None
json.dump(out, open("/workspace/v16_eval/lag_response.json", "w"), indent=1)
print("LAG_DONE", flush=True)
