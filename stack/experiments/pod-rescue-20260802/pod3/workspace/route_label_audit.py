"""Audit the v2 route labels: how often is a REAL turn labelled `straight`?

route_from_future decides TURN vs STRAIGHT from peak_kappa + concentration only.
net_dyaw (the actual net heading change over the 25s route horizon) is computed
and returned but NEVER used in the decision. So a large-radius turn (R>150m)
falls into `is_road` -> ROUTE_STRAIGHT with valid=True, and TRAINS as straight
even if the vehicle turned 90 degrees.

This measures the false-straight rate against net_dyaw as ground truth.
"""
import sys, glob, os, math
import torch

sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
sys.path.insert(0, "/workspace/TanitAD/stack")
import refb_labels as R

VAL = sys.argv[1] if len(sys.argv) > 1 else "/workspace/pai_epcache/physicalai-val-f1b378f295ae"
STRIDE = int(os.environ.get("STRIDE", "20"))
eps = sorted(glob.glob(os.path.join(VAL, "ep_*.pt")))
print(f"episodes: {len(eps)}  horizon={R.NAV_HORIZON_STEPS} steps  min={R.NAV_MIN_STEPS}")
print(f"CURV_TURN={R.CURV_TURN_PER_M:.5f}/m (R={1/R.CURV_TURN_PER_M:.0f}m)  "
      f"CURV_ROAD={R.CURV_ROAD_PER_M:.5f}/m (R={1/R.CURV_ROAD_PER_M:.0f}m)  "
      f"CONC_MIN={R.CONCENTRATION_MIN}")

NAMES = {R.ROUTE_LEFT: "left", R.ROUTE_STRAIGHT: "straight", R.ROUTE_RIGHT: "right"}
BANDS = [(0, 15), (15, 30), (30, 45), (45, 90), (90, 1e9)]
tab = {}          # (band, route, valid) -> count
cls = {}          # route/valid class balance
false_straight = []   # windows: big net_dyaw but labelled straight & VALID
masked_turns = 0
n = 0

for p in eps:
    try:
        d = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  skip {os.path.basename(p)}: {type(e).__name__}"); continue
    poses = d["poses"] if isinstance(d, dict) and "poses" in d else None
    if poses is None:
        continue
    T = poses.shape[0]
    for t in range(0, T, STRIDE):
        r = R.route_from_future(poses, t)
        deg = abs(math.degrees(r["net_dyaw"]))
        band = next(b for b in BANDS if b[0] <= deg < b[1])
        key = (band, r["route"], bool(r["valid"]))
        tab[key] = tab.get(key, 0) + 1
        ck = f"{NAMES[r['route']]}/{'valid' if r['valid'] else ('ambig' if r['ambiguous'] else 'nofuture')}"
        cls[ck] = cls.get(ck, 0) + 1
        n += 1
        if deg >= 30 and r["route"] == R.ROUTE_STRAIGHT:
            if r["valid"]:
                false_straight.append((os.path.basename(p), t, deg,
                                       r["peak_kappa"], r["concentration"]))
            else:
                masked_turns += 1

print(f"\nwindows sampled: {n}")
print("\n=== class balance ===")
for k, v in sorted(cls.items(), key=lambda x: -x[1]):
    print(f"  {k:22s} {v:6d}  {100*v/n:5.1f}%")

print("\n=== |net_dyaw| band  x  assigned label ===")
print(f"  {'band(deg)':>12} {'left':>7} {'straight':>9} {'right':>7} {'INVALID':>9}")
for b in BANDS:
    lo, hi = b
    lbl = f"{lo}-{'inf' if hi > 1e8 else int(hi)}"
    L = tab.get((b, R.ROUTE_LEFT, True), 0)
    S = tab.get((b, R.ROUTE_STRAIGHT, True), 0)
    Rt = tab.get((b, R.ROUTE_RIGHT, True), 0)
    I = sum(v for (bb, _, val), v in tab.items() if bb == b and not val)
    print(f"  {lbl:>12} {L:7d} {S:9d} {Rt:7d} {I:9d}")

tot_big = sum(v for (b, _, _), v in tab.items() if b[0] >= 30)
print(f"\n=== THE BUG, quantified ===")
print(f"  windows with |net_dyaw| >= 30 deg (a REAL turn): {tot_big}")
print(f"  ...labelled STRAIGHT and VALID (trains as straight): {len(false_straight)}"
      f"  = {100*len(false_straight)/max(tot_big,1):.1f}%")
print(f"  ...labelled straight but INVALID (masked, no signal): {masked_turns}"
      f"  = {100*masked_turns/max(tot_big,1):.1f}%")
print(f"  -> genuine turns the strategic head can never learn: "
      f"{100*(len(false_straight)+masked_turns)/max(tot_big,1):.1f}%")

print("\n=== worst false-straights (real turn, labelled straight+valid) ===")
for e, t, deg, pk, cc in sorted(false_straight, key=lambda x: -x[2])[:12]:
    print(f"  {e} t={t:5d}  net_dyaw={deg:6.1f}deg  peak_kappa={pk:.5f} "
          f"(R={1/max(pk,1e-9):7.0f}m)  conc={cc:.2f}")
