"""INDEPENDENT-IMPLEMENTATION cross-check: with the FIXED shared cell, does
`refav1_arm._components`' `LAT_yaw_rate_mae_radps` equal the battery's own A3 cell
(`refcv6_panel.yaw_rate_valid`, written by a different author) on REAL data, and does it
reproduce the battery's banked A3 numbers?

Data: the battery's step-5000 seed-0 early panel on the dev box (read-only).
Code: the battery's `refcv6_panel.cross_paired` (the tip blob, unmodified, from a scratch
copy), with REFCV6_REPO pointing at a clean tip tree whose refav1_arm.py is the FIXED file.
Pairs: read from the battery's banked `raw/a3/cross_paired_A3_step5000_s0_early.json`.

Asserted, per pair: (a) shared cell == A3 cell on every field; (b) A3 cell == the banked A3
cell; (c) every NON-yaw cell == the banked cell (nothing else moved).

usage: python battery_a3_crosscheck.py <battery_code_dir> <panel_dir> <banked_json> <out.json>
"""
from __future__ import annotations

import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
code_dir, panel, banked_path, out_path = sys.argv[1:5]
sys.path.insert(0, code_dir)
import refcv6_panel as P  # noqa: E402

KEYS = ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "n_dropped_nonfinite")
banked = json.load(open(banked_path, encoding="utf-8"))
pairs = []
for nm, blk in banked["pairs"].items():
    if not isinstance(blk, dict) or "direction" not in blk:
        continue
    b_, a_ = [x.strip() for x in blk["direction"].split(" - ")]
    pairs.append((a_, b_, nm))
new = P.cross_paired(panel, pairs, n_boot=2000, seed=0)


def f(c):
    return {k: (c or {}).get(k) for k in KEYS}


rows, bad_a, bad_b, bad_c, n_c = {}, [], [], [], 0
for a_, b_, nm in pairs:
    nb, bb = new["pairs"].get(nm) or {}, banked["pairs"][nm]
    lat_n = (nb.get("families") or {}).get("lateral") or {}
    lat_b = (bb.get("families") or {}).get("lateral") or {}
    sh, va = f(lat_n.get(P.YAW_SHARED)), f(lat_n.get(P.YAW_VALID))
    va_b = f(lat_b.get(P.YAW_VALID))
    rows[nm] = {"direction": bb["direction"], "shared_fixed": sh, "a3_valid": va,
                "a3_valid_banked": va_b,
                "shared_banked_prefix": f(lat_b.get(P.YAW_SHARED))}
    if sh != va:
        bad_a.append(nm)
    if va != va_b:
        bad_b.append(nm)
    for fam, ms in (nb.get("families") or {}).items():
        for mk, c in ms.items():
            if mk in (P.YAW_SHARED, P.YAW_VALID) or not isinstance(c, dict):
                continue
            n_c += 1
            if f(c) != f(((bb.get("families") or {}).get(fam) or {}).get(mk)):
                bad_c.append(f"{nm}/{fam}/{mk}")
out = {"tool": "2026-09-26-yaw-rate-mask/code/probes/battery_a3_crosscheck.py",
       "panel": panel, "n_windows": new["n_windows"], "n_episodes": new["n_episodes"],
       "dt_s": new["dt_s"], "n_pairs": len(pairs),
       "a_shared_equals_a3_every_field": {"pass": not bad_a, "failing_pairs": bad_a},
       "b_a3_reproduces_banked": {"pass": not bad_b, "failing_pairs": bad_b},
       "c_non_yaw_cells_unchanged_vs_banked": {"pass": not bad_c, "n_cells": n_c,
                                               "failing": bad_c[:20]},
       "rows": rows}
json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1)
print(json.dumps({k: out[k] for k in ("n_windows", "n_pairs", "a_shared_equals_a3_every_field",
                                      "b_a3_reproduces_banked",
                                      "c_non_yaw_cells_unchanged_vs_banked")}, indent=1))
r = rows.get("os_minus_refcv4b")
if r:
    print("os - refcv4b: shared(prefix, banked)", r["shared_banked_prefix"])
    print("os - refcv4b: shared(FIXED)        ", r["shared_fixed"])
