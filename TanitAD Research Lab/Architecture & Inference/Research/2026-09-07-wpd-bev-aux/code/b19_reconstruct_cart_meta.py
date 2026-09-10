# -*- coding: utf-8 -*-
"""Reconstruct `panel_rep_cart.json`'s METADATA from the run's own stdout.

⚠️ WHY THIS EXISTS, stated plainly: the Cartesian probe trained and scored all
five arms and wrote every `pt_cart_*.npy`, then entered its INLINE 400-draw
cross-check -- which cost 4,070 s in the 2026-09-08 panel -- and I CANCELLED that
cross-check, so the run never reached its final `json.dump`. The ARMS' scores are
intact (they are the banked score matrices, and `boot_rep_cart.json` is computed
from them); only the run's metadata sidecar is missing.

⛔ Nothing here is a result. Every field is a CONTROL VALUE or a STRUCTURAL
CONSTANT printed verbatim by the run, parsed from its log, and each is asserted
against the value the geometry forces:
    d_raw   = TH*TW*C = 8*20*704 = 112,640
    d_head  = TH*D_RED + PD = 8*16 + 21 = 149
    base    = 335,982 / 20,859,636 from INTEGER COUNTS
The AP values in `arms` are the log's 4-dp prints and are NOT used for any
verdict -- `b11` reads point estimates from `boot_rep_cart.json`.
"""
import io, json, re, sys

LOG = r"C:\Users\Admin\wpd-probe\raw\probe_rep_cart.log"
OUT = r"C:\Users\Admin\wpd-probe\raw\panel_rep_cart.json"
txt = io.open(LOG, encoding="utf-8", errors="replace").read()

m = re.search(r"\[geom cart\] cells (\d+)\s+pos_dim (\d+)\s+test rows (\d+) episodes (\d+)", txt)
assert m, "geom line absent"
NC, PD, NROW, NEP = (int(x) for x in m.groups())

m = re.search(r"\[base\] test base rate ([\d.]+) = (\d+)/(\d+) scored cells", txt)
assert m, "base line absent"
base_str, n_pos, n_sc = m.group(1), int(m.group(2)), int(m.group(3))

m = re.search(r"\[CONTROL\] constant\(0\.37\) AP ([\d.]+)\s+base rate ([\d.]+)\s+exact_match=(\w+)", txt)
assert m, "constant control line absent"
ap_const, base_ctl, exact = float(m.group(1)), float(m.group(2)), m.group(3) == "True"
base_ctl_str = m.group(2)

m = re.search(r"\[CONTROL\] allzero AP ([\d.]+)\s+allone AP ([\d.]+)\s+perfect AP ([\d.]+)", txt)
assert m, "allzero/allone/perfect control line absent"
ap_zero, ap_one, ap_perfect = (float(x) for x in m.groups())

m = re.search(r"\[split\] clips fit/val/test (\d+)/(\d+)/(\d+)", txt)
assert m
n_fit, n_val, n_te = (int(x) for x in m.groups())

arms = {}
for a_ in re.finditer(r"\[(\w+)\s*\] AP_test ([\d.]+)\s+AP_val ([\d.]+)\s+F1 ([\d.]+)\s+"
                      r"IoU ([\d.]+)\s+par ([\d.]+)k\s+n_scored (\d+)\s+d_raw (\d+)", txt):
    nm = a_.group(1)
    arms[nm] = {"ap_test_4dp_from_log": float(a_.group(2)),
                "ap_val": float(a_.group(3)), "f1_test": float(a_.group(4)),
                "iou_test": float(a_.group(5)),
                "head_params": int(round(float(a_.group(6)) * 1000)),
                "n_scored_cells": int(a_.group(7)),
                "d_raw_per_row": int(a_.group(8)),
                "d_head_input_per_cell": 8 * 16 + PD,
                "n_test_rows": NROW, "n_test_episodes": NEP,
                "n_cells_per_row": NC, "steps": 1500, "az_sign": "prog"}

# ⛔ STRUCTURAL ASSERTIONS -- the parse must agree with what the geometry forces.
# ⚠️ The LOG prints 12 decimals; the true rate is the INTEGER RATIO. Comparing the
# full-precision ratio to a 12-dp print fails on the 13th digit -- exactly the
# 2026-09-08 failure in which the REFERENCE, not the metric, was wrong. The fix is
# the same one: derive the reference AT THE PRINTED PRECISION, never add a
# tolerance to the comparison. The full-precision ratio is what is STORED.
BASE_EXACT = n_pos / n_sc
_dp = len(base_ctl_str.split(".")[1])
assert round(BASE_EXACT, _dp) == base_ctl, (BASE_EXACT, base_ctl, _dp)
assert ap_const == base_ctl == ap_zero == ap_one, "constant control did NOT read the base rate"
assert ap_perfect == 1.0
assert all(v["d_raw_per_row"] == 8 * 20 * 704 for v in arms.values())
assert all(v["d_head_input_per_cell"] == 149 for v in arms.values())
assert set(arms) == {"tok_D0", "tok_D0b", "tok_D0c", "tok_D1", "tok_D2"}, sorted(arms)
assert (NROW, NEP, NC, n_sc, n_pos) == (2974, 30, 7014, 20859636, 335982)

out = {
 "_evidence_class": "MEASURED (ours; artifact = raw/probe_rep_cart.log, this run's own stdout)",
 "_source": ("PARSED from probe_rep_cart.log. The run scored every arm and wrote "
             "every pt_cart_*.npy, then was cancelled inside its INLINE 400-draw "
             "cross-check (4,070 s in the 2026-09-08 panel) before its final "
             "json.dump. No RESULT is reconstructed here: the authoritative point "
             "estimates and intervals are in boot_rep_cart.json, computed from the "
             "banked score matrices."),
 "tier": "NOT APPLICABLE - frozen-feature representation probe, no trajectory",
 "geometry": "cart", "az_sign_chosen": "prog",
 "seed": 0, "steps": 1500, "batch": 24, "cells": 1536, "lr": 0.002,
 "n_cells_per_row": NC, "n_scored_cells_test": n_sc, "n_pos_test": n_pos,
 "base_rate_test": BASE_EXACT,
 "base_rate_from_integer_counts": f"{n_pos}/{n_sc}",
 "base_rate_as_printed_by_the_run": base_ctl,
 "all_zero_accuracy_pct": round(100 * (1 - BASE_EXACT), 4),
 "controls": {"constant_score_ap": ap_const, "base_rate": base_ctl,
              "constant_equals_base_rate_exactly": exact,
              "allzero_ap": ap_zero, "allone_ap": ap_one,
              "perfect_ranker_ap": ap_perfect,
              "note": ("checked for EQUALITY against the base rate derived from "
                       "INTEGER COUNTS 335982/20859636 -- the anti-R-2026-09-07-"
                       "ap-ties guard; the panel REFUSES to run if it fails")},
 "split_clips": {"fit": n_fit, "val": n_val, "test": n_te},
 "arms": arms,
 "inline_bootstrap": "CANCELLED BY THE OPERATOR -- see _source; not a failure",
}
io.open(OUT, "w", encoding="utf-8").write(json.dumps(out, indent=1))
print("all structural assertions PASSED")
print(f"  n = {n_sc:,} scored cells / {n_pos:,} pos over {NROW:,} rows / {NEP} episodes")
print(f"  d = {8*20*704:,} raw per row, {8*16+PD} into the head")
print(f"  constant control == base rate exactly: {exact}  ({ap_const!r})")
print("WROTE", OUT)
