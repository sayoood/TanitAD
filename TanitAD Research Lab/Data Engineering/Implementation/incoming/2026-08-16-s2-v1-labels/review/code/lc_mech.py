"""P1 mechanism: the yaw gate does not BOUND the lateral offset above ~7.5 m/s.

Not a statistical claim — an identity. Over a window of arc length L, the
largest lateral offset a CONSTANT-CURVATURE arc can produce while still
passing |net_yaw| <= LC_NET_YAW_MAX is

    lat_max(L) = L * (1 - cos Y) / Y,  Y = LC_NET_YAW_MAX = 0.20 rad

which is 0.0993 * L. It crosses LC_MIN_LAT_M = 3.0 m at L = 30.2 m, i.e. at
v = 7.55 m/s over the 4 s lat horizon. Every aug120 gated clip is faster.
"""
import json
import math
import os

SP = (r"C:\Users\Admin\AppData\Local\Temp\claude"
      r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
      r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
Y = 0.20            # rl.LC_NET_YAW_MAX
LC_MIN = 3.0        # s2_derive.LC_MIN_LAT_M
HALF = 1.75         # rl.LANE_HALF_M
H_S = 4.0           # rl.LAT_HORIZON_STEPS * dt

k = (1 - math.cos(Y)) / Y
print(f"lat_max(L) = {k:.4f} * L   (Y={Y} rad)")
for tgt, name in ((LC_MIN, "LC_MIN_LAT_M=3.0"), (HALF, "LANE_HALF_M=1.75")):
    L = tgt / k
    print(f"  a pure arc reaches {name:>18} at L={L:6.1f} m "
          f"=> v={L/H_S:5.2f} m/s = {L/H_S*3.6:5.1f} km/h")

rows = json.load(open(os.path.join(SP, "lc_feat.json"), encoding="utf-8"))
g = [r for r in rows if r["gated"]]
print(f"\nper-clip table, {len(g)} gated clips "
      f"(PI: 14 wrong / 4 correct / 1 blank)\n")
hdr = (f"{'clip':<10}{'PI':<9}{'tok':<9}{'v':>6}{'L_m':>7}{'net_yaw':>9}"
       f"{'lat_m':>8}{'arc_pred':>10}{'|resid|':>9}{'yawR2':>8}{'bidir':>7}")
print(hdr)
print("-" * len(hdr))
g.sort(key=lambda r: (r.get("pi") or "zz", r["clip_id"]))
n_pred_ge = 0
for r in g:
    f = r["feat"]
    if abs(f["lat_arc_pred"]) >= LC_MIN:
        n_pred_ge += 1
    print(f"{r['clip_id'][:8]:<10}{str(r.get('pi')):<9}{r['ev']['token']:<9}"
          f"{f['v_mean']:>6.1f}{f['L_m']:>7.1f}{f['net_yaw']:>9.3f}"
          f"{f['lat_f']:>8.2f}{f['lat_arc_pred']:>10.2f}"
          f"{f['abs_lat_resid_arc']:>9.2f}{f['yaw_lin_r2']:>8.3f}"
          f"{f['bidir']:>7.3f}")

print(f"\nclips whose PURE-ARC prediction alone already clears the 3.0 m gate: "
      f"{n_pred_ge}/{len(g)}")
same = sum(1 for r in g
           if abs(r['feat']['lat_arc_pred']) >= LC_MIN
           and math.copysign(1, r['feat']['lat_arc_pred'])
           == math.copysign(1, r['feat']['lat_f']))
print(f"  ... of those, same sign as the emitted event: {same}")

# arc-residual gate at the PRE-EXISTING LANE_HALF_M (not tuned here)
lab = [r for r in g if r.get("pi") in ("wrong", "correct")]
for thr in (HALF, LC_MIN):
    tp = sum(1 for r in lab if r["pi"] == "correct"
             and r["feat"]["abs_lat_resid_arc"] >= thr)
    fp = sum(1 for r in lab if r["pi"] == "wrong"
             and r["feat"]["abs_lat_resid_arc"] >= thr)
    print(f"\narc-residual >= {thr}: fires on {tp}/4 CORRECT, {fp}/14 WRONG "
          f"-> would emit {tp+fp}/18, precision "
          f"{(tp/(tp+fp) if tp+fp else float('nan')):.2f}")

# corpus-wide: what would the emission rate be?
print("\n--- corpus-wide arc-residual over ALL 201 aug120 clips ---")
