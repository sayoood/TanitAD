import sys
import numpy as np
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/refe_selection_probe_20261107")
import sel_common as sc
T = np.load(sc.C3 + "/table.npz", allow_pickle=False)
P = T["proposals"].astype(np.float64)
pick = T["pick"].astype(int)
print("proposals", P.shape, "range col2", np.nanmin(P[..., 2]), np.nanmax(P[..., 2]))
xy = np.concatenate([np.zeros_like(P[:, :, :1, :2]), P[:, :, :, :2]], 2)
d = xy[:, :, 1:] - xy[:, :, :-1]
L = np.linalg.norm(d, axis=-1)
ch = np.arctan2(d[..., 1], d[..., 0])
w = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
err = np.abs(w(P[..., 2] - ch))
m = L > 1.0  # at least 1 m per 0.5 s step (2 m/s)
print("moving steps share", m.mean())
print("all proposals: median |col2 - chord| rad per step", np.round([np.median(err[:, :, k][m[:, :, k]]) for k in range(8)], 3))
i = np.arange(len(pick))
ep = err[i, pick]; mp = m[i, pick]
print("picked proposal: median per step", np.round([np.median(ep[:, k][mp[:, k]]) for k in range(8)], 3))
print("picked: p90 per step", np.round([np.percentile(ep[:, k][mp[:, k]], 90) for k in range(8)], 3))
print("picked: share of moving steps with error > 0.3 rad", round(float((ep[mp] > 0.3).mean()), 4))
print("all: share of moving steps with error > 0.3 rad", round(float((err[m] > 0.3).mean()), 4))
b = np.argsort(np.median(np.where(m, err, np.nan)[:, :, :], axis=None) if False else 0) if False else None
per_prop = np.nanmedian(np.where(m, err, np.nan), axis=(0, 2))
print("per-proposal median error, sorted: min/median/max", np.round([np.nanmin(per_prop), np.nanmedian(per_prop), np.nanmax(per_prop)], 3))
c = P[..., 2]
print("col2 vs chord corr (moving, all)", round(float(np.corrcoef(c[m], ch[m])[0, 1]), 4))
print("col2/cumulative check: corr with cumulative yaw of chord at step 8 is not computed; col2 at step 8 median abs", round(float(np.median(np.abs(c[:, :, 7]))), 3))
print("picked col2 first 3 tokens:", np.round(P[i[:3], pick[:3], :, 2], 3).tolist())
print("picked chord first 3 tokens:", np.round(ch[i[:3], pick[:3]], 3).tolist())
print("picked step length first 3:", np.round(L[i[:3], pick[:3]], 2).tolist())
