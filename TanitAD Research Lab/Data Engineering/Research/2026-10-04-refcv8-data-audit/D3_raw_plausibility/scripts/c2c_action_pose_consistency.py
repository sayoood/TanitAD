"""D3 check 2c -- are the two ego channels the model conditions on mutually consistent?
action[:,0] = steer_road_rad = atan(wheelbase * curvature) (physicalai.py:signals_at, legacy wheelbase 2.9); the trainer derives curvature = tan(steer)/2.9.
pose yaw/speed give an INDEPENDENT curvature kappa_pose = dyaw / (vbar * dt) (rad/m).  Rows with vbar > 3 m/s.
Expected: slope(kappa_act on kappa_pose) ~ 1 (the pose channel IS the provider's), corr > 0.99.  Controls: (a) time-shuffled pairing must give corr ~ 0;
(b) a +10 % curvature scaling must read slope 1.10.  Ids sha12."""
import sys, json, hashlib, numpy as np, torch
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]
def run(path):
    m = torch.load(path, map_location="cpu", weights_only=False)
    ka, kp, per = [], [], []
    for cid, P, A in zip(m["clip_id"], m["poses"], m["actions"]):
        P = P.numpy().astype(np.float64); A = A.numpy().astype(np.float64)
        yaw = np.unwrap(P[:, 2]); v = 0.5 * (P[1:, 3] + P[:-1, 3])
        k_pose = np.diff(yaw) / 0.1007 / np.maximum(v, 1e-6)
        k_act = np.tan(0.5 * (A[1:, 0] + A[:-1, 0])) / 2.9
        ok = v > 3.0
        if ok.sum() < 30: continue
        ka.append(k_act[ok]); kp.append(k_pose[ok])
        sl = float(np.polyfit(k_pose[ok], k_act[ok], 1)[0]) if np.std(k_pose[ok]) > 1e-4 else np.nan
        per.append((sha(cid), sl, float(np.std(k_pose[ok]))))
    ka, kp = np.concatenate(ka), np.concatenate(kp)
    sl, ic = np.polyfit(kp, ka, 1); r = float(np.corrcoef(kp, ka)[0, 1])
    rng = np.random.default_rng(0)
    r_sh = float(np.corrcoef(kp, rng.permutation(ka))[0, 1])
    sl10 = float(np.polyfit(kp, 1.10 * ka, 1)[0])
    psl = np.array([p[1] for p in per if p[2] > 5e-3 and not np.isnan(p[1])])
    return {"n_rows": int(len(ka)), "slope": float(sl), "intercept": float(ic), "corr": r, "CONTROL_shuffled_corr": r_sh, "CONTROL_scaled_1.10_slope": sl10,
            "n_clips_with_curvature_range": int(len(psl)), "per_clip_slope_q": [round(float(x), 3) for x in np.quantile(psl, [0.01, 0.05, 0.5, 0.95, 0.99])],
            "n_clips_slope_outside_0.8_1.25": int(((psl < 0.8) | (psl > 1.25)).sum()),
            "worst_sha12": [per[i][0] for i in np.argsort([abs(np.log(max(p[1], 1e-3))) if (p[2] > 5e-3 and not np.isnan(p[1])) else 0 for p in per])[::-1][:5]]}
out = {"train4369": run(sys.argv[1]), "eval139": run("D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139/_v2manifest.pt")}
json.dump(out, open("raw/c2c_action_pose_consistency.json", "w"), indent=1)
print(json.dumps(out, indent=1))
