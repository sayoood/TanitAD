#!/usr/bin/env python
"""⛔ THE FLOORS MUST BE BIT-IDENTICAL ACROSS RUNS, OR THE HARNESS IS NOT HOLDING WHAT IT CLAIMS.

`ha`, `ha0`, `ha0_ext` and `ol` read **no goal, no cost metric and no plan seed** — they are
ego-state replays on the window grid. So two runs over the SAME episodes at the SAME stride must
produce them bit-for-bit, however much their planning arms differ. If a floor moves, every
"the arm beats/loses to the floor" statement in either run is about the harness, not the model.

This compares two dump directories episode by episode and reports, per key:
  * the GT waypoints `g` and the measured `v0` (the inputs the floors are built from),
  * `ha` / `ha0` / `ha0_ext` (the INTEGRATOR form, M11) and `ol`.

⛔ AND IT CARRIES THE SAME-BREATH CONTROL THAT MUST *DIFFER*: `cl`. The two runs differ in
something real (here the cost metric, `cos` vs `ccos`), so `cl` MUST move. An all-equal report
would mean the comparison is reading one file twice, or reading nothing — which on this mount is
indistinguishable from success unless a control fails. *(`grep -c` returning 0 from a file that
could not be read looks exactly like a genuine absence.)*
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

FLOORS = ("ha", "ha0", "ha0_ext", "ol")
INPUTS = ("g", "v0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="dump dir A")
    ap.add_argument("--b", required=True, help="dump dir B")
    ap.add_argument("--control", default="cl",
                    help="an arm that MUST differ between the two runs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    fa = sorted(glob.glob(os.path.join(a.a, "ep*.npz")))
    fb = sorted(glob.glob(os.path.join(a.b, "ep*.npz")))
    n = min(len(fa), len(fb))
    if n == 0:
        raise SystemExit("no overlapping episodes yet")

    out: dict = {"a": a.a, "b": a.b, "n_episodes_a": len(fa),
                 "n_episodes_b": len(fb), "n_compared": n,
                 "ha0_ext_form": "INTEGRATOR (refav1_arm.hold_ext_controls), M11",
                 "episodes": {}}
    all_floors_ok, control_differed = True, False
    for i in range(n):
        za, zb = np.load(fa[i]), np.load(fb[i])
        ca = int(np.asarray(za["clip_index"]).ravel()[0])
        cb = int(np.asarray(zb["clip_index"]).ravel()[0])
        row: dict = {"file": os.path.basename(fa[i]),
                     "same_window_starts": bool((za["ws"] == zb["ws"]).all()),
                     "clip_index_a": ca, "clip_index_b": cb,
                     "same_clip": ca == cb}
        for k in INPUTS + FLOORS:
            if k in za.files and k in zb.files:
                d = float(np.abs(za[k].astype("float64")
                                 - zb[k].astype("float64")).max())
                row[k] = {"max_abs_diff": d, "bit_identical": d == 0.0}
                if k in FLOORS and d != 0.0:
                    all_floors_ok = False
        if a.control in za.files and a.control in zb.files:
            d = float(np.abs(za[a.control].astype("float64")
                             - zb[a.control].astype("float64")).max())
            row[f"{a.control}_CONTROL_must_differ"] = {
                "max_abs_diff": d, "differs": d > 0.0}
            control_differed |= d > 0.0
        all_floors_ok &= row["same_window_starts"] and row["same_clip"]
        out["episodes"][os.path.basename(fa[i])] = row

    out["VERDICT"] = {
        "floors_bit_identical": bool(all_floors_ok),
        "control_arm_differed": bool(control_differed),
        "passes": bool(all_floors_ok and control_differed),
        "reading": ("the floors are the same object in both runs AND the "
                    "comparison can detect a difference"
                    if all_floors_ok and control_differed else
                    "⛔ VOID: either a floor moved (the harness is not holding "
                    "what it claims) or the control did not differ (the "
                    "comparison proves nothing)")}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["VERDICT"], indent=1))
    print(f"[wrote] {a.out}  ({n} episode(s) compared)")
    return 0 if out["VERDICT"]["passes"] else 3


if __name__ == "__main__":
    sys.exit(main())
