import sys, numpy as np, json
from tanitad.data import v8f_mini_data as D
raw = sys.argv[1]
J = D.load_join(raw + "/join_eval141.npz")
z = np.load(raw + "/eval141_kin.npz")
grid = json.load(open(raw + "/eval_grid_windows.json"))
off = np.concatenate([[0], np.cumsum(z["n"])])
print("clips", len(J["clip"]), "N rows", len(J["nav"]), "has", J["has"].mean())
res = []; tokchg = 0; tot = 0; reach0 = 0
for i, c in enumerate(grid["order"]):
    P = z["poses"][off[i]:off[i+1]].astype(np.float64)  # raw 10Hz frames x,y,yaw,v
    o = int(J["off"][i]); n = int(J["n"][i])
    nav = J["nav"][o:o+n]; nd = J["nav_d"][o:o+n]
    xy = P[0::2, :2]
    seg = np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    arc = np.cumsum(seg)
    for t0 in grid["windows"][c]:
        if not (J["has"][o+t0] and np.isfinite(nd[t0]) and nav[t0,:3].argmax() > 0): continue
        tok0 = nav[t0,:3].argmax()
        for t in range(t0+1, min(t0+31, n)):
            tot += 1
            tk = nav[t,:3].argmax()
            if tk != tok0 or not np.isfinite(nd[t]):
                tokchg += 1
                if tk != tok0 and np.isfinite(nd[t0]) and nd[t0] - (arc[t]-arc[t0]) < 5: reach0 += 1
                continue
            res.append((t-t0, nd[t0] - (arc[t]-arc[t0]), nd[t], nd[t0]))
res = np.array(res)
print("pairs (same token, finite)", len(res), "token changed or invalid", tokchg, "of", tot, " of which near-reach", reach0)
err = res[:,2] - res[:,1]
print("nd_logged - (nd0 - arc): mean %.3f  |.| median %.3f  p90 %.3f  max %.3f" % (err.mean(), np.median(np.abs(err)), np.percentile(np.abs(err),90), np.abs(err).max()))
for lo, hi in ((1,5),(6,15),(16,30)):
    m = (res[:,0]>=lo)&(res[:,0]<=hi)
    print("steps %d-%d: n=%d median|err|=%.3f p90=%.3f" % (lo,hi,m.sum(),np.median(np.abs(err[m])),np.percentile(np.abs(err[m]),90)))
print("nd0 range", res[:,3].min(), res[:,3].max(), " nd at t<=0 count", (res[:,2]<=0).sum())
