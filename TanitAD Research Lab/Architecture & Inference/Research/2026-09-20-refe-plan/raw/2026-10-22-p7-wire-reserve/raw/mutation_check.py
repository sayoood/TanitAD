#!/usr/bin/env python3
"""Mutation check for P7-WIRE-RESERVE: each mutation deletes or flips ONE line of the shipped code; the suite (or the stated subset) must go RED, and the unmutated control must be GREEN.

  python mutation_check.py --work <assembled tree: refe/ + raw/> --python <interpreter with numpy> [--only M1,M3] > mutation_check.out

Every mutation asserts that its target text occurs EXACTLY once (a mutation that did not apply would read as 'the suite stayed green').  M1 is the one the brief names: the single
exclusion line of `filter_bank`; it is run against the FULL suite (the sandbox chain tests included).  The rest run the subset that guards them.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FAST = ("UnitTests", "ReserveFileTests", "FilterTests", "AuditTests", "MoveTests", "RealDataTests", "PrepCliTests", "ChainStaticTests")

# id, description, file, old, new, which subset ("FULL" | "FAST" | a --only substring)
MUTATIONS = [
    ("M1", "filter_bank: the single exclusion line deleted (a reserved row is written to the output)", "session_reserve.py",
     "                    continue                                                        # THE EXCLUSION: a reserved row is not written\n", "", "FULL"),
    ("M2", "filter_bank: twins (rank >= 1) are not excluded", "session_reserve.py",
     "                if sess in reserve:\n                    n_exc += 1", "                if sess in reserve and rk == \"0\":\n                    n_exc += 1", "FAST"),
    ("M3", "session_of: accepts any string (the bare rsplit; 'L0' becomes a clean session)", "session_reserve.py",
     "    if not isinstance(log, str) or _LOG_RX.fullmatch(log) is None:\n        raise UnresolvedLog(f\"not a nuPlan log name: {log!r}\")\n    return log.rsplit(\"_\", 2)[0]",
     "    return str(log).rsplit(\"_\", 2)[0]", "FAST"),
    ("M4", "load_reserve: the sha256 pin is not checked", "session_reserve.py",
     "    if got != RESERVE_FILE_SHA256:\n        lf = raw.replace", "    if False:\n        lf = raw.replace", "FAST"),
    ("M5", "audit_inputs: a bank leak is never recorded", "session_reserve.py",
     "        if sess in reserve:\n            leaks.setdefault(inp, []).append(key)", "        if sess in reserve:\n            pass", "FAST"),
    ("M6", "audit_inputs: an unresolved key is not counted", "session_reserve.py",
     "        unres[inp] = unres.get(inp, 0) + 1\n", "        pass\n", "FAST"),
    ("M7", "audit_inputs: an input that examined zero keys counts as clean", "session_reserve.py",
     "    ok = not any(totals.values()) and not any(unres.values()) and not empty", "    ok = not any(totals.values()) and not any(unres.values())", "FAST"),
    ("M8", "move_reserved: the reserve is not moved (left in the shared root)", "session_reserve.py",
     "        dst.parent.mkdir(parents=True, exist_ok=True)\n        shutil.move(str(src), str(dst))\n        n_moved += 1", "        n_moved += 1", "FAST"),
    ("M9", "p6_prep reserve-audit: always prints OK", "p6_prep.py",
     "        if r[\"ok\"]:\n            print(\"ZZRESERVE_AUDIT_OK", "        if True:\n            print(\"ZZRESERVE_AUDIT_OK", "FAST"),
    ("M10", "p6_pod.sh: the resume-bank guard before any move is deleted (a P6 run directory is no longer protected)", "p6_pod.sh",
     "  # THE GUARD, before anything moves: the run in $RUN must have been started on THIS (rebound) bank, or be fresh\n  resume_bank_check\n",
     "  # THE GUARD, before anything moves: the run in $RUN must have been started on THIS (rebound) bank, or be fresh\n", "test_refusals_leave_every_input_untouched"),
    ("M11", "p6_pod.sh: the final audit before ARGS is deleted", "p6_pod.sh",
     "if [ -n \"${P7_RESERVE:-}\" ]; then p7_audit; log \"P7 reserve audit (final): $(tail -n 1 \"$P7D/audit.out\" | cut -c1-160)\"; fi\n", "", "test_a_leak_planted"),
    ("M12", "p6_pod.sh: the reserve block is not guarded by P7_RESERVE (P6's default path would run it)", "p6_pod.sh",
     "if [ -n \"${P7_RESERVE:-}\" ]; then\n  [ -f \"$PKG/refe/session_reserve.py\" ]", "if true; then\n  [ -f \"$PKG/refe/session_reserve.py\" ]", "ChainStaticTests"),
    ("M13", "p6_pod.sh: the bank filter result is not checked (the chain would go on after a refusal)", "p6_pod.sh",
     "    grep -q \"^ZZRESERVE_FILTER_OK \" \"$P7D/filter.out\" || refuse \"P7 reserve: the bank filter refused: $(tail -n 1 \"$P7D/filter.out\" | cut -c1-300)\"\n", "",
     "test_refusals_leave_every_input_untouched"),
]


def run(py, cwd, only, env):
    cmd = [py, "test_session_reserve.py"] + (["--only", only] if only else [])
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env, encoding="utf-8", errors="replace", timeout=3000)
    lines = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.startswith(("ZZSESSRES_", "FAIL:", "ERROR:"))]
    return r.returncode, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8")
    work = Path(a.work)
    want = set(a.only.split(",")) if a.only else None
    bad = 0
    # the controls: the unmutated tree is GREEN on every subset used below
    subsets = {"FAST": None, "FULL": None}
    print("CONTROL (unmutated)")
    for name in sorted({m[5] for m in MUTATIONS if want is None or m[0] in want}):
        if name == "FAST":
            rc = 0
            for cls in FAST:
                r, lines = run(a.python, work / "refe", cls, env)
                rc |= r
                print(f"  FAST/{cls}: rc={r}  {lines[-1] if lines else ''}")
        elif name == "FULL":
            r, lines = run(a.python, work / "refe", None, env)
            rc = r
            print(f"  FULL: rc={r}  {lines[-1] if lines else ''}")
        else:
            r, lines = run(a.python, work / "refe", name, env)
            rc = r
            print(f"  {name}: rc={r}  {lines[-1] if lines else ''}")
        if rc != 0:
            print("  !! the control is not green: the mutation results below mean nothing")
            bad += 1
    print("MUTATIONS (each must be RED)")
    for mid, desc, fname, old, new, where in MUTATIONS:
        if want is not None and mid not in want:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp)
            shutil.copytree(work / "refe", dst / "refe")
            if (work / "raw").is_dir():
                shutil.copytree(work / "raw", dst / "raw")
            f = dst / "refe" / fname
            text = f.read_text(encoding="utf-8")
            if text.count(old) != 1:
                print(f"{mid}  NOT APPLIED ({text.count(old)} matches of the target in {fname}): {desc}")
                bad += 1
                continue
            f.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
            sel = [None] if where == "FULL" else (list(FAST) if where == "FAST" else [where])
            reds, rc_all = [], []
            for s in sel:
                rc, lines = run(a.python, dst / "refe", s, env)
                rc_all.append(rc)
                reds += [ln for ln in lines if ln.startswith(("FAIL:", "ERROR:"))]
                if rc != 0 and where == "FAST":
                    pass
            red = any(rc != 0 for rc in rc_all)
            names = sorted({ln.split("(")[0].replace("FAIL:", "").replace("ERROR:", "").strip() for ln in reds})
            print(f"{mid}  {'RED' if red else 'GREEN  <-- NOT CAUGHT'}  [{where}]  {desc}")
            print(f"      {len(names)} failing test(s): {', '.join(names)[:600]}")
            if not red:
                bad += 1
    print("MUTATION_CHECK_" + ("OK" if bad == 0 else f"FAILED {bad}"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
