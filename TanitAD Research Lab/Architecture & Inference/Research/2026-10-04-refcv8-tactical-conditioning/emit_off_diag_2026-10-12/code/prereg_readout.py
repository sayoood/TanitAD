"""Apply PREREG_EMIT_OFF.md's readout rules MECHANICALLY to diag_describe.py outputs. The rules are copied here as code
and must not be edited after the full-grid numbers exist.
usage: python prereg_readout.py <arm> <emitoff_vs_own_base.json> <emitoff_vs_V0_base.json> <own_base_vs_V0_base.json>
Prints the outcome (A / B / C) and, for every metric, the gap fraction closed (arm_base - emit_off) / (arm_base - V0)."""
import json
import sys

PRIMARY = "cross_abs_6s"
SHOW = ("cross_abs_6s", "curv_mae_0_2s", "yaw_mae_0_2s", "head15_all", "dir_correct_turn", "ade_all", "ade_straight",
        "off_drivable", "speed_mae_0_2s", "speed_mae_2_6s", "ceil_violation", "lat3_f1_v9", "map_iou_drivable",
        "map_iou_lane")
arm = sys.argv[1]
eo_own, eo_v0, own_v0 = (json.load(open(p, encoding="utf-8")) for p in sys.argv[2:5])
seeds = ("s0", "s1")


def cell(d, s, m):
    return d["seeds"][s]["pair"].get(m) or {}


a_better_own = all(cell(eo_own, s, PRIMARY).get("sep_better") for s in seeds)
a_not_worse_v0 = all(not cell(eo_v0, s, PRIMARY).get("sep_worse") for s in seeds)
b_worse_v0 = all(cell(eo_v0, s, PRIMARY).get("sep_worse") for s in seeds)
b_not_better_own = all(not cell(eo_own, s, PRIMARY).get("sep_better") for s in seeds)
if a_better_own and a_not_worse_v0:
    outcome = "A -- EMISSION makes it"
elif b_worse_v0 and b_not_better_own:
    outcome = "B -- TRAINING makes it"
else:
    outcome = "C -- MIXED"
print(f"{arm}: OUTCOME {outcome}")
print(f"  rule inputs on {PRIMARY}: emit_off better than own base (both seeds) = {a_better_own}; "
      f"emit_off not worse than V0 (both) = {a_not_worse_v0}; emit_off worse than V0 (both) = {b_worse_v0}; "
      f"emit_off not better than own base (both) = {b_not_better_own}")
print(f"  {'metric':20s} {'seed':4s} {'V0':>8s} {'arm':>8s} {'emit_off':>8s} {'gap closed':>10s}  eo-vs-own   eo-vs-V0")
for m in SHOW:
    for s in seeds:
        c_ov, c_eo, c_e0 = cell(own_v0, s, m), cell(eo_own, s, m), cell(eo_v0, s, m)
        if not (c_ov and c_eo):
            continue
        v0, armb, eo = c_ov["base"], c_ov["treat"], c_eo["treat"]
        gap = armb - v0
        frac = (armb - eo) / gap if abs(gap) > 1e-12 else float("nan")
        tag = lambda c: "BETTER" if c.get("sep_better") else ("WORSE" if c.get("sep_worse") else "ns")
        print(f"  {m:20s} {s:4s} {v0:8.4f} {armb:8.4f} {eo:8.4f} {frac:10.2f}  {tag(c_eo):10s} {tag(c_e0):10s}")
