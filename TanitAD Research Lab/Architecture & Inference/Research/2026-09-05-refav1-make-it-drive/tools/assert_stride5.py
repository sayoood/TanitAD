#!/usr/bin/env python
"""The control the stride-5 extraction never got.

⭐ WHY. `logits_stride40.npz` was extracted with `--assert-dump`, so its argmax
is proven equal to a banked dump's `goal_lat_cl` on 282/282 windows — it is
anchored. `logits_stride5.npz` (n = 1974) was extracted WITHOUT that flag, and
this package used it to claim the bias ladder is "stable on a 7x wider panel".
An unanchored panel cannot support that claim.

The anchor is available for free: a stride-40 grid is a SUBSET of a stride-5
grid (40 = 8 x 5), so every stride-40 window must appear in the stride-5 file
with IDENTICAL logits. This asserts exactly that, joining on (clip_index, ws).

⛔ CONTROLS:
  * coverage first — if the stride-40 windows are not FOUND in the stride-5
    file, the comparison is vacuous and this reports INCONCLUSIVE, not PASS;
  * a DELIBERATE MISMATCH control: the same comparison against stride-5 rows
    shifted by one window must FAIL, or "identical" says nothing about the join;
  * v0 and the labels are compared too, not just the logits — a matching logit
    row under a mismatched v0 would be a different window.

Run: python assert_stride5.py --s40 <npz> --s5 <npz> --out <json>
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s40", required=True)
    ap.add_argument("--s5", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A = np.load(a.s40, allow_pickle=False)
    B = np.load(a.s5, allow_pickle=False)
    kb = {(int(c), int(w)): i for i, (c, w) in
          enumerate(zip(B["clip_index"], B["ws"]))}
    idx = [kb.get((int(c), int(w)))
           for c, w in zip(A["clip_index"], A["ws"])]
    found = [i for i in idx if i is not None]
    n40 = len(idx)

    R = {"s40": a.s40, "s5": a.s5, "n_s40": int(n40), "n_s5": int(len(B["ws"])),
         "n_found_in_s5": len(found),
         "coverage": len(found) / n40 if n40 else 0.0, "checks": {}}
    if len(found) != n40:
        R["PASS"] = False
        R["verdict"] = ("INCONCLUSIVE — not every stride-40 window is present "
                        "in the stride-5 file; the join, not the logits, is "
                        "what failed")
        json.dump(R, open(a.out, "w"), indent=1)
        print(json.dumps(R, indent=1))
        return 2

    j = np.array(found)
    fails = []

    def chk(name, ok, detail):
        R["checks"][name] = {"ok": bool(ok), **detail}
        if not ok:
            fails.append(name)

    for key in ("lat_logits", "lon_logits", "route_logits"):
        d = float(np.abs(A[key] - B[key][j]).max())
        chk(f"{key}_bit_identical", bool(np.array_equal(A[key], B[key][j])),
            {"max_abs_diff": d})
    for key in ("v0", "lat_label", "nav_cmd"):
        chk(f"{key}_identical", bool(np.array_equal(A[key], B[key][j])),
            {"n": int(len(j))})
    chk("argmax_identical",
        bool(np.array_equal(A["lat_logits"].argmax(-1),
                            B["lat_logits"][j].argmax(-1))),
        {"n": int(len(j))})

    # DELIBERATE-MISMATCH CONTROL: shift the join by one window. If THIS also
    # reads identical, the comparison is not discriminating anything.
    js = (j + 1) % len(B["ws"])
    same_shifted = bool(np.array_equal(A["lat_logits"], B["lat_logits"][js]))
    frac_shift_eq = float((A["lat_logits"].argmax(-1) ==
                           B["lat_logits"][js].argmax(-1)).mean())
    chk("CONTROL_shifted_join_is_NOT_identical", not same_shifted,
        {"argmax_agreement_when_shifted": frac_shift_eq,
         "note": "must be False/low, else the join proves nothing"})

    R["PASS"] = not fails
    R["failed_checks"] = fails
    R["verdict"] = ("the stride-5 panel is ANCHORED: it reproduces the "
                    "assert-dump-controlled stride-40 extraction exactly on "
                    "every shared window" if R["PASS"] else "FAILED")
    json.dump(R, open(a.out, "w"), indent=1)
    print(json.dumps(R, indent=1))
    print("\nOVERALL:", "PASS" if R["PASS"] else f"FAIL {fails}")
    return 0 if R["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
