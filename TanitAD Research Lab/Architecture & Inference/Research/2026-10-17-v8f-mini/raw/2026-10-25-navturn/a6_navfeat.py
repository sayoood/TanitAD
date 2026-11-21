"""a6_navfeat.py -- is 'route silent' (token FOLLOW) really silent? The nav FEATURE vector T reads is (one-hot token, d_next/100 clipped [0,3], args_valid) (v8f_mini_data.py:203-206);
build_v9_labels.py:272-278 sets args_valid=1 and d_next for the nearest announced turn EVEN when the token stays FOLLOW (d_next > d_ann). MEASURED on eval-141 at the row T reads (rowT).
Also prints per-cause SSE shares of the 210. EXPLORATORY.
"""
import csv
import numpy as np
from nt_common import *

J, K, T, grid = load_inputs()
rows = list(csv.DictReader(open(HERE + "navturn_windows.csv", encoding="utf-8")))
N = len(T["ci"]); ci = T["ci"]; rowT = T["rowT"]; G = T["gt_std"]; has = T["has"]
CONS = J["cons"].astype(np.float64)
cols = list(range(1, 10))
cause = np.array([""] * N, dtype=object); typ = np.array([""] * N, dtype=object)
for r in rows:
    cause[int(r["w"])] = r["cause"]; typ[int(r["w"])] = r["type"]
A = typ == "A"
navT = J["nav"][rowT]
valid = navT[:, 4] == 1
dnav = navT[:, 3] * 100.0
dlab = CONS[rowT, 3]
same = A & valid & (np.abs(dnav - dlab) <= 1.5)
diff = A & valid & ~same
inval = A & ~valid
print(f"A (banked 210) by what the nav FEATURES at the T-tick row say (token is FOLLOW in all):")
print(f"  args_valid = 0 (no announced turn ahead in the whole recording): {int(inval.sum())}")
print(f"  args_valid = 1, d_next == label d_start (|diff| <= 1.5 m, SAME event, the token just has not fired): {int(same.sum())}")
print(f"  args_valid = 1, d_next != label d_start (a DIFFERENT, later announced turn): {int(diff.sum())}")


def sq(arm):
    r = T[f"T_{arm}"] - G
    return np.where(np.isfinite(r), r ** 2, 0.0), np.isfinite(r)


def rmse_ci(E, P, m, B=2000, seed=0):
    ue = np.unique(ci); idx = [np.where(ci == e)[0] for e in ue]
    num = np.where(m, E[:, cols].sum(1), 0.0); den = np.where(m, P[:, cols].sum(1), 0.0)
    sn = np.array([num[i].sum() for i in idx]); sd = np.array([den[i].sum() for i in idx])
    rg = np.random.default_rng(seed); out = np.empty(B)
    for b in range(B):
        p = rg.integers(0, len(ue), len(ue)); out[b] = sn[p].sum() / max(sd[p].sum(), 1e-12)
    return np.sqrt(num.sum() / max(den.sum(), 1)), np.sqrt(np.percentile(out, 2.5)), np.sqrt(np.percentile(out, 97.5))


for arm in ("M2b_s0", "M2b_s1"):
    E, P = sq(arm)
    print(f"\n{arm}: turn-cell RMSE (SD units) [95% ep-boot], windows, by nav-feature state")
    for nm, m in (("valid, SAME event", same), ("valid, different event", diff), ("invalid", inval), ("all 210", A)):
        if m.sum():
            r, lo, hi = rmse_ci(E, P, m)
            print(f"  {nm:24s} n {int(m.sum()):4d} (episodes {len(np.unique(ci[m])):3d})  RMSE {r:.3f} [{lo:.3f}, {hi:.3f}]")
    print("  by cause (of the 210): share of the 210's turn-cell SSE, nav state counts (same/diff/invalid)")
    tot = E[A][:, cols].sum()
    for c in ("CURVE", "SUPPRESSED", "HORIZON", "UNRESOLVED"):
        m = A & (cause == c)
        print(f"    {c:11s} SSE share {E[m][:, cols].sum() / tot:6.1%}   nav: same {int((m & same).sum()):3d}  diff {int((m & diff).sum()):3d}  invalid {int((m & inval).sum()):3d}")
# FOLLOW windows overall
tokT = navT[:, :3].argmax(1)
allf = has & (tokT == 0)
print(f"\nALL FOLLOW windows at rowT: {int(allf.sum())}; args_valid=1 in {int((allf & valid).sum())} ({(allf & valid).sum() / allf.sum():.1%}); join-wide (all rows with a v9 row): "
      f"{int(((J['nav'][:, :3].argmax(1) == 0) & (J['nav'][:, 4] == 1) & J['has']).sum())} of {int(((J['nav'][:, :3].argmax(1) == 0) & J['has']).sum())}")
