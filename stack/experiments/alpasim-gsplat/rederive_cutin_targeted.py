#!/usr/bin/env python3
"""ADVERSARIAL independent re-derivation of the cut-in-targeted panel.

Re-implemented from the RAW rollout json only. Does NOT import cutin_targeted.py.
Checks the raw recorded scalar v_target (no metric-definition ambiguity) and the
window/cluster denominators.
"""
import json, sys
from pathlib import Path
import numpy as np

R = Path("G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack/experiments/alpasim-gsplat/results/scene2-cutin-targeted")

def load(name):
    return json.loads((R / "rollouts" / f"rollouts_{name}.json").read_text())

def wall_ticks(d):
    ts = np.asarray([g["ts_us"] for g in d["gt"]], float)
    out = {}
    for r in d["rollouts"]:
        s = int(r["start_frame"])
        for st in r["steps"]:
            out[(s, int(st["k"]))] = int(np.argmin(np.abs(ts - float(st["t_us"]))))
    return out

def field(d, key):
    out = {}
    for r in d["rollouts"]:
        s = int(r["start_frame"])
        for st in r["steps"]:
            out[(s, int(st["k"]))] = float(st[key])
    return out

# exposure ticks: renderable crossing 118..143, +20 post ticks  (POST_TICKS=20)
V = json.loads((R / "CUTIN_IS_REAL.json").read_text())
T_EXP, T_SHAM = set(), set()
for c in V["per_crossing"]:
    tgt = T_EXP if c["renderable"] else T_SHAM
    for k in range(c["k_start"], c["k_end"] + 20 + 1):
        tgt.add(k)
T_SHAM -= T_EXP
print("MY T_EXP  n=%d range=[%d,%d]" % (len(T_EXP), min(T_EXP), max(T_EXP)))
print("MY T_SHAM n=%d range=[%d,%d]" % (len(T_SHAM), min(T_SHAM), max(T_SHAM)))

def paired_cluster_boot(dA, dB, clusters, n_boot=4000, seed=20260803):
    """Paired episode-cluster bootstrap: resample CLUSTERS with replacement,
    take the mean of (A-B) over all rows in the resampled clusters."""
    rng = np.random.default_rng(seed)
    cl = sorted(set(clusters))
    idx = {c: np.where(np.asarray(clusters) == c)[0] for c in cl}
    diff = np.asarray(dA) - np.asarray(dB)
    point = float(diff.mean())
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(cl, size=len(cl), replace=True)
        rows = np.concatenate([idx[c] for c in pick])
        stats[b] = diff[rows].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi), bool(lo > 0 or hi < 0)

def run(armA, armB, label):
    dA, dB = load(armA), load(armB)
    wA, wB = wall_ticks(dA), wall_ticks(dB)
    common = sorted(set(wA) & set(wB))
    # exogenous clock must agree across the two files
    disagree = [c for c in common if wA[c] != wB[c]]
    print(f"\n===== {label}   common (start,k) rows = {len(common)}  "
          f"clock disagreements = {len(disagree)}")
    exp = [c for c in common if wA[c] in T_EXP]
    sham = [c for c in common if wA[c] in T_SHAM]
    rest = [c for c in common if wA[c] not in T_EXP and wA[c] not in T_SHAM]
    print("  n_all=%d cl=%d | n_exposure=%d cl=%d | n_sham=%d cl=%d | n_other=%d cl=%d" % (
        len(common), len({c[0] for c in common}),
        len(exp), len({c[0] for c in exp}),
        len(sham), len({c[0] for c in sham}),
        len(rest), len({c[0] for c in rest})))
    for key in ("v_target", "v"):
        fA, fB = field(dA, key), field(dB, key)
        for tag, keys in (("EXPOSURE", exp), ("ALL", common), ("SHAM", sham)):
            if not keys: continue
            a = [fA[c] for c in keys]; b = [fB[c] for c in keys]
            cl = [c[0] for c in keys]
            p, lo, hi, sep = paired_cluster_boot(a, b, cl)
            print("  %-9s %-9s pooled  d=%+8.4f [%+7.4f,%+7.4f] sep=%-5s n=%d cl=%d"
                  % (tag, key, p, lo, hi, sep, len(keys), len(set(cl))))
    # k bands, both denominators
    fA, fB = field(dA, "v_target"), field(dB, "v_target")
    for tag, keys in (("ALLwin", common), ("EXPwin", exp)):
        for lo_k, hi_k in ((0,5),(5,13),(13,26),(26,38),(38,50)):
            kk = [c for c in keys if lo_k <= c[1] < hi_k]
            if not kk: continue
            a = [fA[c] for c in kk]; b = [fB[c] for c in kk]
            cl = [c[0] for c in kk]
            p, lo, hi, sep = paired_cluster_boot(a, b, cl)
            print("  %-6s k[%2d,%2d) v_target d=%+8.4f [%+7.4f,%+7.4f] sep=%-5s n=%-4d cl=%d"
                  % (tag, lo_k, hi_k, p, lo, hi, sep, len(kk), len(set(cl))))

run("flagship-v1_objects", "flagship-v1_empty", "FLAGSHIP objects - empty")
run("refc-base_objects", "refc-base_empty", "REFC objects - empty")
#!/usr/bin/env python3
"""Part 2: the claims that have NO banked artifact -- EARLY/LATE, leave-one-cluster-out --
plus the stratum x k-band cross-tab that shows what 'k 0-5 on ALL windows' really contains.
"""
import json
from pathlib import Path
import numpy as np

R = Path("G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack/experiments/alpasim-gsplat/results/scene2-cutin-targeted")
load = lambda n: json.loads((R / "rollouts" / f"rollouts_{n}.json").read_text())

def wall_ticks(d):
    ts = np.asarray([g["ts_us"] for g in d["gt"]], float)
    return {(int(r["start_frame"]), int(st["k"])): int(np.argmin(np.abs(ts - float(st["t_us"]))))
            for r in d["rollouts"] for st in r["steps"]}

def field(d, key):
    return {(int(r["start_frame"]), int(st["k"])): float(st[key])
            for r in d["rollouts"] for st in r["steps"]}

V = json.loads((R / "CUTIN_IS_REAL.json").read_text())
T_EXP, T_SHAM = set(), set()
for c in V["per_crossing"]:
    (T_EXP if c["renderable"] else T_SHAM).update(range(c["k_start"], c["k_end"] + 21))
T_SHAM -= T_EXP

def boot(dA, dB, clusters, n_boot=4000, seed=20260803):
    rng = np.random.default_rng(seed)
    cl = sorted(set(clusters)); arr = np.asarray(clusters)
    idx = {c: np.where(arr == c)[0] for c in cl}
    diff = np.asarray(dA) - np.asarray(dB)
    st = np.empty(n_boot)
    for b in range(n_boot):
        rows = np.concatenate([idx[c] for c in rng.choice(cl, size=len(cl), replace=True)])
        st[b] = diff[rows].mean()
    lo, hi = np.percentile(st, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0)

dA, dB = load("flagship-v1_objects"), load("flagship-v1_empty")
rA, rB = load("refc-base_objects"), load("refc-base_empty")
w = wall_ticks(dA)
common = sorted(w)

print("### CROSS-TAB: what fraction of each k-band is an EXPOSURE window?")
print("  k-band   n_all  n_exposure  n_sham  n_other   %exposure")
for lo_k, hi_k in ((0,5),(5,13),(13,26),(26,38),(38,50)):
    kk = [c for c in common if lo_k <= c[1] < hi_k]
    ne = sum(w[c] in T_EXP for c in kk)
    ns = sum(w[c] in T_SHAM for c in kk)
    print("  k[%2d,%2d) %5d %10d %7d %8d %9.1f%%"
          % (lo_k, hi_k, len(kk), ne, ns, len(kk)-ne-ns, 100*ne/len(kk)))

print("\n### EARLY (starts 75-84) vs LATE (starts 87-109), k 0-5, ALL windows")
for arm, (A, B) in (("flagship", (dA, dB)), ("refc", (rA, rB))):
    for key in ("v_target", "v"):
        fA, fB = field(A, key), field(B, key)
        for tag, sel in (("EARLY 75-84", lambda s: s <= 84), ("LATE 87-109", lambda s: s >= 87),
                         ("ALL 12", lambda s: True)):
            kk = [c for c in common if 0 <= c[1] < 5 and sel(c[0])]
            p, lo, hi, sep = boot([fA[c] for c in kk], [fB[c] for c in kk], [c[0] for c in kk])
            print("  %-8s %-9s %-12s d=%+8.4f [%+7.4f,%+7.4f] sep=%-5s n=%-3d cl=%d"
                  % (arm, key, tag, p, lo, hi, sep, len(kk), len({c[0] for c in kk})))

print("\n### LEAVE-ONE-CLUSTER-OUT, flagship v_target, k 0-5, ALL windows")
fA, fB = field(dA, "v_target"), field(dB, "v_target")
base = [c for c in common if 0 <= c[1] < 5]
starts = sorted({c[0] for c in base})
pts = []
for drop in starts:
    kk = [c for c in base if c[0] != drop]
    p, lo, hi, sep = boot([fA[c] for c in kk], [fB[c] for c in kk], [c[0] for c in kk])
    pts.append(p)
    print("  drop %3d  d=%+8.4f [%+7.4f,%+7.4f] sep=%s" % (drop, p, lo, hi, sep))
print("  LOO range: %+.4f .. %+.4f" % (min(pts), max(pts)))

print("\n### DROP THE 4 EARLY CLUSTERS (75,78,81,84) -- the open_risk's own test")
kk = [c for c in base if c[0] >= 87]
p, lo, hi, sep = boot([fA[c] for c in kk], [fB[c] for c in kk], [c[0] for c in kk])
print("  flagship v_target k0-5 without 75-84: d=%+.4f [%+.4f,%+.4f] sep=%s n=%d cl=%d"
      % (p, lo, hi, sep, len(kk), len({c[0] for c in kk})))

print("\n### PER-START mean v_target diff at k 0-5 (is it 4 clusters or all 12?)")
for s in starts:
    kk = [c for c in base if c[0] == s]
    d = np.mean([fA[c] - fB[c] for c in kk])
    print("  start %3d  mean d=%+8.4f  (n=%d)" % (s, d, len(kk)))
#!/usr/bin/env python3
"""Part 3: is 'EARLY starts' the same fact as 'sham epoch'? And the residual
endogeneity of the METRIC's reference index (i_gt), which the selector fix did not touch.
"""
import json
from pathlib import Path
import numpy as np

R = Path("G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack/experiments/alpasim-gsplat/results/scene2-cutin-targeted")
load = lambda n: json.loads((R / "rollouts" / f"rollouts_{n}.json").read_text())
V = json.loads((R / "CUTIN_IS_REAL.json").read_text())
T_EXP, T_SHAM = set(), set()
for c in V["per_crossing"]:
    (T_EXP if c["renderable"] else T_SHAM).update(range(c["k_start"], c["k_end"] + 21))
T_SHAM -= T_EXP

dA, dB = load("flagship-v1_objects"), load("flagship-v1_empty")
ts = np.asarray([g["ts_us"] for g in dA["gt"]], float)

print("### start -> wall tick at k=0, and which stratum the k 0-5 rows fall in")
print("  start  wt(k=0)  stratum(k=0)   k0-5: nEXP nSHAM nOTHER   group")
for r in sorted(dA["rollouts"], key=lambda x: x["start_frame"]):
    s = int(r["start_frame"])
    wts = [int(np.argmin(np.abs(ts - float(st["t_us"])))) for st in r["steps"][:5]]
    strat = lambda t: "EXP" if t in T_EXP else ("SHAM" if t in T_SHAM else "OTHER")
    ne = sum(t in T_EXP for t in wts); ns = sum(t in T_SHAM for t in wts)
    print("   %3d    %3d      %-6s          %d    %d     %d       %s"
          % (s, wts[0], strat(wts[0]), ne, ns, 5-ne-ns,
             "EARLY(75-84)" if s <= 84 else "LATE(87-109)"))

print("\n### Does i_gt (the METRIC's gt reference) differ between conditions at k 0-5?")
print("   -- the selector was made exogenous, but `ade` and `abs_target_speed_err`")
print("      still score the plan against gt[i_gt], and i_gt IS post-treatment.")
igA = {(int(r["start_frame"]), int(st["k"])): int(st["i_gt"])
       for r in dA["rollouts"] for st in r["steps"]}
igB = {(int(r["start_frame"]), int(st["k"])): int(st["i_gt"])
       for r in dB["rollouts"] for st in r["steps"]}
wt = {(int(r["start_frame"]), int(st["k"])): int(np.argmin(np.abs(ts - float(st["t_us"]))))
      for r in dA["rollouts"] for st in r["steps"]}
for lo_k, hi_k in ((0,5),(5,13),(13,26),(26,38),(38,50)):
    kk = [c for c in igA if lo_k <= c[1] < hi_k]
    d = np.array([igA[c] - igB[c] for c in kk])
    dw = np.array([igA[c] - wt[c] for c in kk])
    print("   k[%2d,%2d)  i_gt_A - i_gt_B: mean %+6.2f  median %+5.1f  n_nonzero %3d/%3d"
          "  |  i_gt_A - wall: median %+5.1f" % (lo_k, hi_k, d.mean(), np.median(d),
                                                  int((d != 0).sum()), len(d), np.median(dw)))

print("\n### gv[i_gt] -- the LONGITUDINAL target the arms are scored against")
gv = None
# reconstruct log speed from gt poses (finite difference), same as cl_metrics would
xy = np.array([[g["x"], g["y"]] for g in dA["gt"]])
tt = ts / 1e6
sp = np.zeros(len(xy))
sp[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(np.diff(tt), 1e-6)
sp[0] = sp[1]
for lo_k, hi_k in ((0,5),(38,50)):
    kk = [c for c in igA if lo_k <= c[1] < hi_k]
    a = np.array([sp[igA[c]] for c in kk]); b = np.array([sp[igB[c]] for c in kk])
    print("   k[%2d,%2d)  log speed at i_gt_A mean %.3f  at i_gt_B mean %.3f  diff %+0.4f m/s"
          % (lo_k, hi_k, a.mean(), b.mean(), (a - b).mean()))
