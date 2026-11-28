"""D-REPLICATE-FPRATE / H-ESTIM-SEED-1 under the FIXED yaw cell.

Recurses every arm's `paired_vs_A0` in the banked withheld-bank `panel_report.json` exactly as
`2026-09-06-register-repair/raw/replicate_fp_rate.py` does (every dict carrying `delta` and
`separated` is one cell), then substitutes the yaw cells re-derived with the fixed
`refav1_arm._components` (`raw/claims/wbank_panel.result.json`, whose reproduction control read
0 differences on 300 cells). Nothing else is recomputed: the fix moves only the yaw cells (0 of the
other cells differ, same artifact).

Controls: C1 the A0b block is non-empty (42 cells); C2 the banked counts reproduce the register's
"6 / 42" before any substitution.

usage: python replicate_fp_rate_fixed.py <panel_report.json> <wbank_panel.result.json> <out.json>
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
panel = json.load(open(sys.argv[1], encoding="utf-8"))
res = json.load(open(sys.argv[2], encoding="utf-8"))
YAW = "LAT_yaw_rate_mae_radps"


def cells(o, path=()):
    if isinstance(o, dict):
        if "separated" in o and "delta" in o:
            yield path, o
        else:
            for k, v in o.items():
                yield from cells(v, path + (k,))


fixed = {tuple(k.split(" | ")): v for k, v in res["yaw_fixed"].items()}   # (arm, tag, sec, mk)
out = {"tool": "2026-09-26-yaw-rate-mask/code/probes/replicate_fp_rate_fixed.py", "arms": {}}
for arm, rec in panel["arms"].items():
    pv = rec.get("paired_vs_A0")
    if not pv:
        continue
    n = n_sep_b = n_sep_f = 0
    yaw_b = yaw_f = 0
    two_s_rows_b = two_s_rows_f = 0
    two_s_n = 0
    seven = {"LON_speed_mae_mps", "LON_along_mae_m", "LON_accel_mae_mps2", "LAT_cross_mae_m",
             "LAT_heading_mae_deg", YAW, "ade_m"}
    for path, c in cells(pv):
        n += 1
        sb = bool(c["separated"])
        sf = sb
        if path[-1] == YAW:
            key = (arm, path[0], path[1], YAW)
            fv = fixed[key]
            sf = bool(fv[3])
            yaw_b += sb
            yaw_f += sf
        n_sep_b += sb
        n_sep_f += sf
        if len(path) == 3 and path[1] == "plan_2s" and path[2] in seven and path[0] in ("k", "w"):
            two_s_n += 1
            two_s_rows_b += sb
            two_s_rows_f += sf
    out["arms"][arm] = {"n_cells": n, "separated_banked": n_sep_b, "separated_fixed": n_sep_f,
                        "yaw_cells_separated_banked": yaw_b, "yaw_cells_separated_fixed": yaw_f,
                        "rate_banked": round(n_sep_b / n, 4), "rate_fixed": round(n_sep_f / n, 4),
                        "verdict_view_2s_7rows": {"n": two_s_n, "banked": two_s_rows_b,
                                                  "fixed": two_s_rows_f}}
a0b = out["arms"]["A0b_replicate"]
out["C1_block_non_empty"] = a0b["n_cells"] == 42
out["C2_banked_reproduces_6_of_42"] = a0b["separated_banked"] == 6
json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), indent=1)
for arm, r in out["arms"].items():
    print(f"{arm:14s} cells {r['n_cells']:3d}  separated {r['separated_banked']:2d} -> {r['separated_fixed']:2d}"
          f"  (yaw {r['yaw_cells_separated_banked']} -> {r['yaw_cells_separated_fixed']})"
          f"  rate {r['rate_banked']:.4f} -> {r['rate_fixed']:.4f}"
          f"  | VERDICT 2s 7-row view {r['verdict_view_2s_7rows']}")
print("C1", out["C1_block_non_empty"], "C2", out["C2_banked_reproduces_6_of_42"])
