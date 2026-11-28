"""a3_error.py -- what the nav/turn-label disagreement costs T, by cause; and what a consistent definition could (and could not) move (EXPLORATORY, MEASURED eval-141).

Reads navturn_windows.csv (a2_classify.py) and the banked table.npz. T-error units: SD of the standardiser, present cells only (R63), turn cells = cells 1..9 (as s2c_routesilent.py).
Estimator: episode-cluster bootstrap B=2000 over the 141 episodes (the resampling of taniteval/ci.py, implemented here as in the T_ATTRIB scripts); it answers 'another draw of EPISODES' only.
The (E) outer bound is the T_ATTRIB definition: sum over the stratum of (own ADE6 - GT-constraints ADE6) / 2,399, T0 tier (O fed the GT code).
"""
import csv
import numpy as np
from nt_common import *

J, K, T, grid = load_inputs()
rows = list(csv.DictReader(open(HERE + "navturn_windows.csv", encoding="utf-8")))
N = len(T["ci"])
ci, tt = T["ci"], T["tt"]
has = T["has"]
G = T["gt_std"]
cause = np.array([""] * N, dtype=object)
typ = np.array([""] * N, dtype=object)
for r in rows:
    cause[int(r["w"])] = r["cause"]
    typ[int(r["w"])] = r["type"]
isturn = {int(r["w"]): r["is_turn"] for r in rows}
cols = list(range(1, 10))
rowN = J["off"][ci] + tt
tokN = J["nav"][rowN][:, :3].argmax(1)
labT = np.isfinite(G[:, 1]) & has
A = typ == "A"


def sq(arm):
    r = T[f"T_{arm}"] - G
    return np.where(np.isfinite(r), r ** 2, 0.0), np.isfinite(r)


def boot_ratio(num, den, eid, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    ue = np.unique(eid)
    idx = [np.where(eid == e)[0] for e in ue]
    sn = np.array([num[i].sum() for i in idx]); sd_ = np.array([den[i].sum() for i in idx])
    out = np.empty(B)
    for b in range(B):
        p = rng.integers(0, len(ue), len(ue))
        out[b] = sn[p].sum() / max(sd_[p].sum(), 1e-12)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def rmse_ci(E, P, mask, eid, B=2000, seed=0):
    num = np.where(mask, E[:, cols].sum(1), 0.0)
    den = np.where(mask, P[:, cols].sum(1), 0.0)
    pt = np.sqrt(num.sum() / max(den.sum(), 1))
    lo, hi = boot_ratio(num, den, eid, B, seed)
    return pt, float(np.sqrt(lo)), float(np.sqrt(hi)), int(den.sum())


# ---------------------------------------------------------------- 1. SSE by cause (share of ALL T squared error, banked phi definition)
causes_A = ["CURVE", "SUPPRESSED", "HORIZON", "UNRESOLVED"]
for arm in ("M2b_s0", "M2b_s1"):
    E, P = sq(arm)
    tot_all = E[has].sum()
    tot_turn = E[labT][:, cols].sum()
    m210 = A
    print(f"\n== {arm}: total T squared error (all present cells, {int(has.sum())} windows) = {tot_all:.1f}; turn cells (1..9) in the 210 = {E[m210][:, cols].sum():.1f} -> phi = {E[m210][:, cols].sum() / tot_all:.3f} "
          f"(banked 0.189 / 0.181); share of all turn-cell SSE (turn-label windows): {E[m210][:, cols].sum() / tot_turn:.3f}")
    print(f"  {'group':22s} {'n win':>5s} {'turn-cell n':>11s} {'RMSE':>6s} [95% ep-boot]   {'phi (share of all T SSE)':>25s}  [CI]")
    for name, m in [("A: all 210", A)] + [(f"A: {c}", A & (cause == c)) for c in causes_A]:
        if m.sum() == 0:
            continue
        pt, lo, hi, ncell = rmse_ci(E, P, m, ci)
        num = np.where(m, E[:, cols].sum(1), 0.0); den = np.where(has, E.sum(1), 0.0)
        l2, h2 = boot_ratio(num, den, ci)
        print(f"  {name:22s} {int(m.sum()):5d} {ncell:11d} {pt:6.3f} [{lo:.3f}, {hi:.3f}]   {num.sum() / den.sum():25.3f}  [{l2:.3f}, {h2:.3f}]")
    spk = labT & (tokN != 0)
    pt, lo, hi, ncell = rmse_ci(E, P, spk, ci)
    print(f"  {'route SPEAKS (banked)':22s} {int(spk.sum()):5d} {ncell:11d} {pt:6.3f} [{lo:.3f}, {hi:.3f}]")
    # like-for-like: silent but is_turn (HORIZON + SUPPRESSED, REPRO-decided) vs speaking
    sil_it = A & np.isin(cause, ["HORIZON", "SUPPRESSED"])
    pt, lo, hi, ncell = rmse_ci(E, P, sil_it, ci)
    print(f"  {'silent AND is_turn':22s} {int(sil_it.sum()):5d} {ncell:11d} {pt:6.3f} [{lo:.3f}, {hi:.3f}]   <- like-for-like with 'route SPEAKS' (both junction-type events)")
    sil_c = A & (cause == "CURVE")
    pt2, lo2, hi2, nc2 = rmse_ci(E, P, sil_c, ci)
    print(f"  {'silent AND curve':22s} {int(sil_c.sum()):5d} {nc2:11d} {pt2:6.3f} [{lo2:.3f}, {hi2:.3f}]   <- every one has vmin > 8 m/s; NO speaking window can have vmin > 8 (is_turn gate)")
    # per-cell RMSE: curve vs speaking vs silent is_turn
    print("  per-cell RMSE (SD units):  cell                 CURVE   silent-is_turn   SPEAKS")
    for q in cols:
        def cr(m):
            mm = m & np.isfinite(G[:, q])
            return np.sqrt(E[mm, q].sum() / max(mm.sum(), 1)), int(mm.sum())
        a, b, c = cr(sil_c), cr(sil_it), cr(spk)
        print(f"                             {CONS_NAMES[q]:18s} {a[0]:6.3f}   {b[0]:6.3f}          {c[0]:6.3f}   (n {a[1]}/{b[1]}/{c[1]})")

# ---------------------------------------------------------------- 2. (E) outer bound per cause: sum (own ADE6 - GT-code ADE6) / 2399
print("\n== (E) EXACT window-stratum outer bound on ADE6 (T0): sum over stratum (own - GT-constraints ADE6) / 2,399; assumes T's constraints PERFECT in those windows")
for arm in ("M2b_s0", "M2b_s1"):
    gap = np.load(TA + f"gap_{arm}.npz")["gc"].astype(np.float64)      # the banked s7_ceilings definition: own ADE6 - ADE6 with the GT constraint block (gap_cons)
    print(f"  {arm}: gap_cons {gap.sum() / N:.4f} m (mean over {N} windows; banked gap_cons file gap_{arm}.npz)")
    for name, m in [("A: all 210", A)] + [(f"A: {c}", A & (cause == c)) for c in causes_A] + [("B: all 129", typ == "B")] + [(f"B: {c}", (typ == 'B') & (cause == c)) for c in sorted(set(cause[typ == 'B']))]:
        if m.sum():
            print(f"    {name:22s} n {int(m.sum()):4d}  (E) {gap[m].sum() / N:7.4f} m  ({gap[m].sum() / gap.sum():6.1%} of the T->GT gap)   mean per-window gap {gap[m].mean():.3f} m vs all-window mean {gap.mean():.3f}")
print("  bar 2F = 0.2415 m (T_ATTRIB)")

# ---------------------------------------------------------------- 3. counterfactual estimates for the 210 (turn cells), s0 / s1
print("\n== ESTIMATED removable turn-cell squared error in the 210 under each consistent definition (HYPOTHESIS where marked)")
for arm in ("M2b_s0", "M2b_s1"):
    E, P = sq(arm)
    tot_all = E[has].sum()
    spk = labT & (tokN != 0)
    r_spk = np.sqrt(E[spk][:, cols].sum() / P[spk][:, cols].sum())
    sse = {c: E[A & (cause == c)][:, cols].sum() for c in causes_A}
    ncell = {c: P[A & (cause == c)][:, cols].sum() for c in causes_A}
    s210 = sum(sse.values())
    print(f"  {arm}: turn-cell SSE in the 210 = {s210:.1f} (phi {s210 / tot_all:.3f}); route-speaks RMSE = {r_spk:.3f}")
    lab_side = sse["CURVE"] + sse["SUPPRESSED"]
    print(f"    (L) LABEL-side: null the turn cells of non-announced events (CURVE + SUPPRESSED = {int(A.sum() and (A & np.isin(cause, ['CURVE', 'SUPPRESSED'])).sum())} windows): metric loses {lab_side:.1f} = {lab_side / s210:.1%} of the 210's turn-cell SSE = phi {lab_side / tot_all:.3f}. "
          f"This moves the METRIC, not the driving; it also removes the (possibly useful) curve-geometry supervision.")
    nav_isturn = (sse["HORIZON"] - ncell["HORIZON"] * r_spk ** 2) + (sse["SUPPRESSED"] - ncell["SUPPRESSED"] * r_spk ** 2)
    print(f"    (N1) NAV-side, like-for-like: let the nav speak for the {int((A & np.isin(cause, ['HORIZON', 'SUPPRESSED'])).sum())} silent is_turn windows (longer lookahead / no suppression): "
          f"error to the route-speaks level frees {max(nav_isturn, 0):.1f} = {max(nav_isturn, 0) / s210:.1%} of the 210's SSE = phi {max(nav_isturn, 0) / tot_all:.3f}  [ESTIMATED; needs the nav to supply the ego's own future path = optimistic]")
    nav_curve = sse["CURVE"] - ncell["CURVE"] * r_spk ** 2
    print(f"    (N2) NAV-side, announce CURVES too: if T reached the junction-turn error on curves, frees {max(nav_curve, 0):.1f} = {max(nav_curve, 0) / s210:.1%} of the 210's SSE = phi {max(nav_curve, 0) / tot_all:.3f}  "
          f"[HYPOTHESIS: the 0.40 level was measured on slow tight junction turns only; and it hands T the future path of the ego]")

# ---------------------------------------------------------------- 4. echo / information-disjointness check
print("\n== nav token vs the situation class (information-disjointness; join rows at rowT)")
rowT = T["rowT"]
lc = J["lat_cls"][rowT]
navT = J["nav"][rowT][:, :3].argmax(1)
for k, nm in ((1, "TURN_L"), (2, "TURN_R")):
    m = has & (lc == k)
    print(f"  lat_cls {nm}: n {int(m.sum())}; nav token = same side in {int((m & (navT == k)).sum())}; nav FOLLOW in {int((m & (navT == 0)).sum())}; other side {int((m & (navT == 3 - k)).sum())}")
m = has & (lc == 0)
print(f"  lat_cls LANE_KEEP: n {int(m.sum())}; nav TURN in {int((m & (navT != 0)).sum())} ({(m & (navT != 0)).sum() / m.sum():.1%})")
m = has & (navT != 0)
print(f"  nav TURN windows (rowT) n {int(m.sum())}: lat_cls TURN same side {int((m & (lc == navT)).sum())}; LANE_KEEP {int((m & (lc == 0)).sum())}; none {int((m & (lc == -100)).sum())}")
print(f"  turn-label cells present: with lat_cls TURN_L/R {int((labT & np.isin(lc, [1, 2])).sum())}, LANE_KEEP {int((labT & (lc == 0)).sum())}, none {int((labT & (lc == -100)).sum())}  (of {int(labT.sum())})")
