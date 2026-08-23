"""S-curve probe, dump half (Sayed's idea, 2026-08-06): a yaw-rate SIGN REVERSAL inside
the 2 s window cannot be extrapolated from ego dynamics at t0 — reproducing it implies
reading the environment (or the future actions; the hold-action GPU arm settles which,
queued behind run 6).

S-window (GT): net heading change of the plan's first half and second half have OPPOSITE
signs and both exceed THR. Ego-only baselines by construction: CTR (hold the entry yaw
rate -> h1,h2 same sign, can never reproduce an S) and CV (straight).

Per arm: both-lobe sign agreement on S-windows, second-lobe magnitude ratio, false-S rate
on non-S windows (an always-wiggling arm would fake reversals), ADE on S vs non-S.
"""
import glob, json
import numpy as np

DT, THR = 0.1, 0.03           # rad; ~1.7 deg per half-window
dumps = sorted(glob.glob("v16dump/ep*.npz"))
assert len(dumps) == 40

def halves(P):
    p = np.concatenate([np.zeros((P.shape[0], 1, 2)), P[..., :2]], 1)
    d = p[:, 1:] - p[:, :-1]
    h = np.arctan2(d[..., 1], d[..., 0])
    dh = (h[:, 1:] - h[:, :-1] + np.pi) % (2 * np.pi) - np.pi
    ok = (np.linalg.norm(d, axis=-1)[:, 1:] > 0.05) & (np.linalg.norm(d, axis=-1)[:, :-1] > 0.05)
    dh = np.where(ok, dh, 0.0)
    return dh[:, :9].sum(1), dh[:, 9:].sum(1), dh

res = {a: {"hit": [], "fake": [], "ratio": [], "ade_s": [], "ade_ns": []}
       for a in ("v1arch", "v16", "ctr", "cv")}
n_s = n_ns = 0
per_ep_hit = {a: [] for a in res}
for f in dumps:
    d = np.load(f)
    G = d["g"].astype(np.float64)
    g1, g2, gdh = halves(G)
    is_s = (np.sign(g1) != np.sign(g2)) & (np.abs(g1) > THR) & (np.abs(g2) > THR)
    n_s += int(is_s.sum()); n_ns += int((~is_s).sum())
    v0 = np.linalg.norm(G[:, 0, :2], axis=1) / DT
    yr0 = gdh[:, 0] / DT                     # entry yaw rate (ego-observable at t0)
    arcs = {}
    t = np.arange(1, 21) * DT
    th = yr0[:, None] * t[None, :]
    r = np.where(np.abs(yr0) > 1e-4, v0 / np.where(np.abs(yr0) > 1e-4, yr0, 1.0), 0.0)
    ctr = np.zeros_like(G[..., :2])
    m = np.abs(yr0) > 1e-4
    ctr[m, :, 0] = r[m, None] * np.sin(th[m]); ctr[m, :, 1] = r[m, None] * (1 - np.cos(th[m]))
    ctr[~m, :, 0] = v0[~m, None] * t[None, :]
    cv = np.zeros_like(ctr); cv[:, :, 0] = v0[:, None] * t[None, :]
    for key, P in (("v1arch", d["a"].astype(np.float64)), ("v16", d["b"].astype(np.float64)),
                   ("ctr", ctr), ("cv", cv)):
        p1, p2, _ = halves(P)
        hit = (np.sign(p1) == np.sign(g1)) & (np.sign(p2) == np.sign(g2)) \
              & (np.abs(p1) > THR / 2) & (np.abs(p2) > THR / 2)
        fake = (np.sign(p1) != np.sign(p2)) & (np.abs(p1) > THR) & (np.abs(p2) > THR)
        res[key]["hit"] += list(hit[is_s]); res[key]["fake"] += list(fake[~is_s])
        res[key]["ratio"] += list((p2[is_s] / g2[is_s]))
        ade = np.linalg.norm(P[..., :2] - G[..., :2], axis=-1).mean(1)
        res[key]["ade_s"] += list(ade[is_s]); res[key]["ade_ns"] += list(ade[~is_s])
        per_ep_hit[key].append(float(hit[is_s].mean()) if is_s.sum() else np.nan)

out = {"_def": f"S = GT half-window net headings opposite signs, both>|{THR}| rad; "
               "hit = arm matches BOTH lobe signs (mag>thr/2); ctr = hold entry yaw rate "
               "(ego-only; structurally cannot S); n from 6,834 stride-1 windows",
       "n_s_windows": n_s, "n_non_s": n_ns, "s_fraction": round(n_s / (n_s + n_ns), 4)}
rng = np.random.default_rng(0)
for a in res:
    hits = np.array(res[a]["hit"], dtype=float)
    eh = np.array([x for x in per_ep_hit[a] if np.isfinite(x)])
    draws = [float(eh[rng.integers(0, len(eh), len(eh))].mean()) for _ in range(2000)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    out[a] = {"s_reproduction_rate": round(float(hits.mean()), 4),
              "rate_ci95_epboot": [round(float(lo), 4), round(float(hi), 4)],
              "second_lobe_ratio_median": round(float(np.median(res[a]["ratio"])), 4),
              "false_s_rate": round(float(np.mean(res[a]["fake"])), 4),
              "ade_on_s": round(float(np.mean(res[a]["ade_s"])), 4),
              "ade_on_non_s": round(float(np.mean(res[a]["ade_ns"])), 4)}
    print(a, json.dumps(out[a]))
json.dump(out, open("s_curve_dump.json", "w"), indent=1)
print("n_S", n_s, "of", n_s + n_ns)
