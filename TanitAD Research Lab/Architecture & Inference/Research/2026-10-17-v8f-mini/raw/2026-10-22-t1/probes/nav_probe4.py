import sys, numpy as np, json
from tanitad.data import v8f_mini_data as D
raw = sys.argv[1]
J = D.load_join(raw + "/join_eval141.npz"); z = np.load(raw + "/eval141_kin.npz"); grid = json.load(open(raw + "/eval_grid_windows.json"))
off = np.concatenate([[0], np.cumsum(z["n"])])
tokc = np.bincount(J["nav"][:, :3].argmax(1), minlength=3); print("token counts F/L/R", tokc)
valid = J["nav"][:, 4] > 0
print("valid by token:", [int(((J["nav"][:, :3].argmax(1) == k) & valid).sum()) for k in range(3)])
errs = []; chg = 0; tot = 0
for i, c in enumerate(grid["order"]):
    P = z["poses"][off[i]:off[i+1]].astype(np.float64); o = int(J["off"][i]); n = int(J["n"][i])
    nav, nd = J["nav"][o:o+n], J["nav_d"][o:o+n]; xy = P[0::2, :2]; arc = np.cumsum(np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1)])
    for t0 in grid["windows"][c]:
        if not (J["has"][o+t0] and np.isfinite(nd[t0]) and nav[t0, :3].argmax() == 0): continue
        for t in range(t0+1, min(t0+31, n)):
            tot += 1
            if nav[t, :3].argmax() != 0 or not np.isfinite(nd[t]): chg += 1; continue
            errs.append(nd[t] - max(nd[t0] - (arc[t]-arc[t0]), 0.0))
e = np.abs(np.array(errs)); print("FOLLOW+valid windows: pairs", tot, "changed/invalid", chg, "| held: n=%d median|e|=%.4f p99=%.2f frac<1m=%.4f" % (len(e), np.median(e) if len(e) else -1, np.percentile(e,99) if len(e) else -1, (e<1).mean() if len(e) else -1))
