"""violation_geometry.py -- DESCRIPTIVE (no bar): what own-map accuracy would candidate SELECTION need?  Seed-0 MAPSEL trace + rasters, every 4th window
with GT.  (a) where (x) and how DEEP (penetration into the GT sidewalk, EDT) the GT-sidewalk hits of the fan candidates are; (b) the per-window AUC of the
own map's gated (x >= 4 m) max sidewalk probability at separating GT-violating from GT-clean candidates, beside the same score on GT itself and a
deranged (next-episode) own map.  CPU, 0 GPU."""
import sys, os
import numpy as np, torch
from scipy.ndimage import distance_transform_edt
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_mapsel/code")
import mapsel_score as MS
L = MS.L
tr = MS.load_trace("F:/refcv9_mapsel/trace_s0.npz")
R = "F:/refcv9_mapsel/rast_s0"
don = MS.donor_table(tr)
X_GATE = 4.0
dep, xs, own_hit, own_near = [], [], [], []
auc_own, auc_der = [], []
nwin = 0
def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or (~y).all():
        return None
    r = np.argsort(np.argsort(s)) + 1.0
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
for j in range(0, tr["n"], 4):
    f = MS.rast_file(R, tr["e_i"][j], tr["ws"][j])
    z = np.load(f)
    if "gt" not in z.files or don[j] < 0:
        continue
    dz = np.load(MS.rast_file(R, tr["e_i"][don[j]], tr["ws"][don[j]]))
    gt, psw, dpsw = z["gt"], z["psw"].astype(np.float32) / 255, dz["psw"].astype(np.float32) / 255
    sw = (gt == 7)
    edt = distance_transform_edt(sw) * 0.1
    near = torch.from_numpy(np.asarray(psw >= 0.5))
    g = L.Geom(torch.from_numpy(tr["fan"][j]))
    ix, iy, ins = g.ix.numpy(), g.iy.numpy(), g.inside.numpy()
    x = ix * 0.1
    hit = ins & sw[ix, iy]
    viol = hit.reshape(hit.shape[0], -1).any(1)
    nwin += 1
    if viol.any():
        d = edt[ix, iy][hit]; dep.append(d); xs.append(x[hit]); own_hit.append((psw[ix, iy] >= 0.5)[hit])
    m = ins & (x >= X_GATE)
    so = np.where(m, psw[ix, iy], 0).reshape(m.shape[0], -1).max(1)
    sd = np.where(m, dpsw[ix, iy], 0).reshape(m.shape[0], -1).max(1)
    a1, a2 = auc(so, viol), auc(sd, viol)
    if a1 is not None:
        auc_own.append(a1); auc_der.append(a2)
dep, xs, oh = np.concatenate(dep), np.concatenate(xs), np.concatenate(own_hit)
print(f"windows {nwin}; GT-sidewalk footprint hits {len(dep)}; windows with mixed violating/clean candidates {len(auc_own)}")
print("penetration depth quantiles (m) p25/p50/p75/p90:", np.round(np.quantile(dep, [.25, .5, .75, .9]), 2).tolist())
for a, b in ((0, 4), (4, 10), (10, 20), (20, 40), (40, 100)):
    m = (xs >= a) & (xs < b)
    print(f"x {a:>3}-{b:<3} m: share of hits {m.mean():.3f}; median depth {np.median(dep[m]) if m.any() else float('nan'):.2f} m; own flags the hit cell {oh[m].mean() if m.any() else float('nan'):.3f}")
print(f"per-window AUC (violating vs clean candidates; score = max own psw on footprint at x >= {X_GATE} m): OWN mean {np.mean(auc_own):.3f}, DERANGED mean {np.mean(auc_der):.3f}, n {len(auc_own)}")
