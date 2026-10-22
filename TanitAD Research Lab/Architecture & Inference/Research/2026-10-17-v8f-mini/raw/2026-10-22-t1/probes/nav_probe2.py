import sys, numpy as np, json
from tanitad.data import v8f_mini_data as D
raw = sys.argv[1]
J = D.load_join(raw + "/join_eval141.npz")
z = np.load(raw + "/eval141_kin.npz")
grid = json.load(open(raw + "/eval_grid_windows.json"))
off = np.concatenate([[0], np.cumsum(z["n"])])
errs = []; cls = {"held_d_pos": 0, "held_d_zero": 0, "chg_to_follow": 0, "chg_to_other": 0, "invalid_same_tok": 0}
firstzero = []
for i, c in enumerate(grid["order"]):
    P = z["poses"][off[i]:off[i+1]].astype(np.float64)
    o = int(J["off"][i]); n = int(J["n"][i])
    nav = J["nav"][o:o+n]; nd = J["nav_d"][o:o+n]
    xy = P[0::2, :2]; arc = np.cumsum(np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1)])
    for t0 in grid["windows"][c]:
        if not (J["has"][o+t0] and np.isfinite(nd[t0]) and nav[t0,:3].argmax() > 0): continue
        tok0 = nav[t0,:3].argmax()
        for t in range(t0+1, min(t0+31, n)):
            tk = nav[t,:3].argmax()
            dr = max(nd[t0] - (arc[t]-arc[t0]), 0.0)
            if tk == tok0 and np.isfinite(nd[t]):
                errs.append(nd[t] - dr)
                cls["held_d_zero" if nd[t] == 0 else "held_d_pos"] += 1
            elif tk == tok0:
                cls["invalid_same_tok"] += 1
            else:
                cls["chg_to_follow" if tk == 0 else "chg_to_other"] += 1
                firstzero.append(dr)
errs = np.array(errs)
print(cls)
print("clamped dead-reckon error: n=%d median|e|=%.4f p90=%.3f p99=%.3f max=%.2f frac|e|<1m=%.4f" % (len(errs), np.median(np.abs(errs)), np.percentile(np.abs(errs),90), np.percentile(np.abs(errs),99), np.abs(errs).max(), (np.abs(errs)<1).mean()))
fz = np.array(firstzero)
print("token-change events: n=%d, dead-reckoned remaining distance at the change: median %.2f p10 %.2f p90 %.2f frac(dr==0)=%.3f" % (len(fz), np.median(fz), np.percentile(fz,10), np.percentile(fz,90), (fz==0).mean()))
