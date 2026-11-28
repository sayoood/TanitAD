"""s10_ade_deltas.py -- paired ADE6 deltas of each lever arm vs M2b_s0 (episode-cluster bootstrap B=2000), the window-coverage facts, and the panel-slope prediction vs realised ADE
for the arms with a separated SPEED move. MEASURED on eval-141; own-code ADE6 (T1-class open loop). EXPLORATORY."""
import numpy as np
from tcommon import *
T = load(); eid = T["ci"]; ue = np.unique(eid); has = T["has"]
print("no-v9-row windows:", int((~has).sum()), "in episodes", sorted(set(eid[~has].tolist())), "; episodes with a row:", len(set(eid[has].tolist())))
rng = np.random.default_rng(0)
boots = rng.integers(0, len(ue), (2000, len(ue)))
cnt = np.array([(eid == e).sum() for e in ue])
def pair(a, b):
    d = a - b
    s = np.array([d[eid == e].sum() for e in ue])
    bs = np.array([s[k].sum() / cnt[k].sum() for k in boots])
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
print(f"{'arm vs M2b_s0':14s} {'dADE6 m':>8s} {'[CI95]':>18s}")
for a in M2B_FAMILY[1:]:
    m, lo, hi = pair(T[f"ade6_{a}"], T["ade6_M2b_s0"])
    print(f"{a:14s} {m:+8.4f} [{lo:+.4f},{hi:+.4f}] {'separated' if lo > 0 or hi < 0 else 'ns'}")
# panel-slope prediction for SPEED
from s3_tail import KIND
sp = [i for i in KIND if KIND[i].startswith("SPEED")]
E0, P = sq(T, "M2b_s0"); E0[~has] = 0
base = E0[:, sp].sum() / has.sum()
print("\nSPEED-kind SSE per window (SD^2), M2b_s0:", round(float(base), 4))
for a, slope in (("pgt_s0", 0.8999), ("wc_s0", 0.8999), ("df50_s0", 0.8999), ("hist10_s0", 0.8999), ("grid8x20_s0", 0.8999)):
    Ea = sq(T, a)[0]; Ea[~has] = 0
    d_sse = Ea[:, sp].sum() / has.sum() - base
    m, lo, hi = pair(T[f"ade6_{a}"], T["ade6_M2b_s0"])
    print(f"   {a:12s} dSPEED SSE/window {d_sse:+.4f} -> panel-slope (0.90 m/SD^2) predicts dADE6 {slope * d_sse:+.3f} m; realised {m:+.3f} [{lo:+.3f},{hi:+.3f}]")
