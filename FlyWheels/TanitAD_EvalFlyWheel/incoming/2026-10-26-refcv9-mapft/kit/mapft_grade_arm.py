"""mapft_grade_arm.py -- grade ONE RED arm of the refcv8-mapft launch gate (PREREG_MAPFT s1) from the arm run's EVIDENCE files.

Why not from the arm's TOKEN (the grader this replaces): `launch_gate.finalize` rejects every evidence file that carries an `arm`
("an arm run can never PASS"), so an arm token's `checks` read MISSING BY CONSTRUCTION and a token-based grader can never see a
catch. MEASURED 2026-10-27 01:15 on C3 a0c93c84: both arms FAILed G-DVB and G-LIVE with map-only reasons of their own (planner
arm: 186 tensors trainable outside the map modules, 175 frozen tensors changed by step 50; map-frozen arm: 18 map tensors frozen,
12 map groups with no gradient), both were bound to the main token's commit / tree / argv / profile, the main token read PASS on
both checks -- and the old grader printed ESCAPED-OR-MISCREDITED for both, i.e. "the freeze guard is decorative", which was false.

CAUGHT iff ALL of:
  * the arm token exists, is not PASS, and names THIS arm as the active regression arm;
  * for at least one of G-DVB / G-LIVE, the arm run's evidence file is THIS check and THIS arm, is bound to the MAIN token's
    commit / tree / argv / profile, reads FAIL or ERROR, and carries a map-only reason of its own;
  * the arm-free MAIN token (the CONTROL) is PASS and reads PASS on every such check.
Prints one line; the substring `-CAUGHT` appears in it only on a catch (`mapft_prelaunch.sh fire` greps for it).

usage: python mapft_grade_arm.py <arm> <arm run dir> <main token path>
"""
from __future__ import annotations

import glob
import json
import os
import sys

CHECKS = ("G-DVB", "G-LIVE")
BIND_KEYS = ("argv_sha256", "commit", "profile", "tree_sha256")


def _load(p):
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def token_of(pattern):
    t = None
    for f in sorted(glob.glob(pattern)):
        j = _load(f)
        if isinstance(j, dict) and j.get("kind") == "launch-gate-token":
            t = j
    return t


def grade(arm, d, main):
    tok, ctl = token_of(os.path.join(d, "*.json")), token_of(main)
    if tok is None or ctl is None:
        return f"ARM-{arm}-UNGRADED(no arm token or no control token)"
    mb, cc = ctl.get("binding") or {}, ctl.get("checks") or {}
    named = any(f"a regression arm is active ({arm})" in str(r) for r in tok.get("reasons") or [])
    red, sig, notes = [], [], []
    for c in CHECKS:
        ev = _load(os.path.join(d, "evidence", c + ".json"))
        if not isinstance(ev, dict):
            notes.append(f"{c}:no-evidence")
            continue
        b = ev.get("binding") or {}
        if ev.get("check") != c or ev.get("arm") != arm:
            notes.append(f"{c}:not-this-arm")
            continue
        if any(k not in b for k in BIND_KEYS) or any(mb.get(k) != v for k, v in b.items()):
            notes.append(f"{c}:binding-differs")
            continue
        if ev.get("status") in ("FAIL", "ERROR"):
            red.append(c)
            if any("map-only" in str(r).lower() for r in ev.get("reasons") or []):
                sig.append(c)
    ctl_ok = ctl.get("verdict") == "PASS" and bool(sig) and all((cc.get(c) or {}).get("status") == "PASS" for c in sig)
    caught = tok.get("verdict") != "PASS" and named and bool(sig) and ctl_ok
    return (f"ARM-{arm}-{'CAUGHT' if caught else 'ESCAPED-OR-MISCREDITED'} verdict={tok.get('verdict')} arm_named={named} "
            f"red_checks={','.join(red)} map_only_reason_on={','.join(sig)} control_pass={ctl_ok} source=evidence"
            + (f" notes={','.join(notes)}" if notes else ""))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: mapft_grade_arm.py <arm> <arm run dir> <main token path>")
    print(grade(*sys.argv[1:]))
