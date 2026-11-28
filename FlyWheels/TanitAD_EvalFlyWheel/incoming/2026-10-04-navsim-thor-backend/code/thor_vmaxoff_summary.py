#!/usr/bin/env python3
"""Summarise the refcv7 LEGAL row (R7_VMAXOFF) from THOR scores with the suite's OWN parser (TANITAD VENV).

Runs ``…/refcv7-standard-tests/navsim/code/vmaxoff_legal_summary7.py`` UNCHANGED on a STAGED root in a
scratch directory, so no file is ever written under the milestone's standard names:
  <stage>/vmaxoff_legal/scores_navtest/r7s50400_R7_VMAXOFF/r7s50400_R7_VMAXOFF.{csv,counts.json}  <- Thor
  <stage>/scores_navtest/r7s50400_R7_A1{,_s1}/...                                               <- Thor, if banked PASS
  (navhard) <stage>/vmaxoff_legal/scores_navhard/score_R7_VMAXOFF__navhard_two_stage.{csv,counts.json} + _wrapper/frame,
            <stage>/vmaxoff_legal/bridge_navhard/seam_R7_VMAXOFF.{npz,manifest.json}            <- Thor merge + dev-box seam
Floors (STOP / CV / HUMAN navtest; STOP_zero / CV_official / ECHO navhard) are the parser's own dev-box
banks => every VMAXOFF-vs-floor pair is MIXED-BACKEND and admissible only under RULING_BACKEND_POLICY.md.
The parser's output JSON is copied to ``--dest`` as ``summary_<split>_vmaxoff_legal.THOR.json`` with a
``backend`` block added.

    python thor_vmaxoff_summary.py --split navtest --thor-nt <dev-box thor_scores/navtest> --stage <scratch> --dest <thor_scores/navtest>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess

SUITE = "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"


def lp(p: str) -> str:
    """Windows MAX_PATH guard (MEASURED: the merged navhard frame path is > 260 chars and open() raised
    FileNotFoundError on a file that exists)."""
    if os.name == "nt":
        p = os.path.abspath(p)
        prefix = "\\\\?\\"
        if not p.startswith(prefix):
            p = prefix + p
    return p


def _cp_pass(src_csv: str, src_counts: str, dst_dir: str, dst_stem: str) -> bool:
    try:
        st = json.load(open(src_counts, encoding="utf-8")).get("status")
    except (OSError, ValueError):
        return False
    if st != "PASS" or not os.path.exists(src_csv):
        return False
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copyfile(src_csv, os.path.join(dst_dir, dst_stem + ".csv"))
    shutil.copyfile(src_counts, os.path.join(dst_dir, dst_stem + ".counts.json"))
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("navtest", "navhard"), required=True)
    ap.add_argument("--thor", required=True, help="dev-box thor_scores/<split> dir")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--legal-dir", default=f"{SUITE}/raw/milestones/step50400/vmaxoff_legal")
    a = ap.parse_args()
    if os.path.isdir(a.stage):
        shutil.rmtree(a.stage)
    staged = {}
    if a.split == "navtest":
        for arm, dst in (("R7_VMAXOFF", os.path.join(a.stage, "vmaxoff_legal", "scores_navtest")),
                         ("R7_A1", os.path.join(a.stage, "scores_navtest")),
                         ("R7_A1_s1", os.path.join(a.stage, "scores_navtest"))):
            lab = f"r7thor_s50400_{arm}"
            staged[arm] = _cp_pass(os.path.join(a.thor, lab, f"{lab}.csv"), os.path.join(a.thor, lab, f"{lab}.counts.json"),
                                   os.path.join(dst, f"r7s50400_{arm}"), f"r7s50400_{arm}")
    else:
        sfx = "__navhard_two_stage"
        for arm, dst in (("R7_VMAXOFF", os.path.join(a.stage, "vmaxoff_legal", "scores_navhard")),
                         ("R7_A1", os.path.join(a.stage, "scores_navhard")), ("R7_A1_s1", os.path.join(a.stage, "scores_navhard"))):
            src = os.path.join(a.thor, f"{arm}_THOR")
            ok = _cp_pass(os.path.join(src, f"score_{arm}_THOR{sfx}.csv"), os.path.join(src, f"score_{arm}_THOR{sfx}.counts.json"),
                          dst, f"score_{arm}{sfx}")
            if ok:
                w = os.path.join(dst, f"score_{arm}{sfx}_wrapper")
                os.makedirs(w, exist_ok=True)
                shutil.copyfile(lp(os.path.join(src, f"score_{arm}_THOR{sfx}_wrapper", f"{arm}_THOR{sfx}_final_scores_frame.csv")),
                                os.path.join(w, f"{arm}{sfx}_final_scores_frame.csv"))
            staged[arm] = ok
        bsrc = os.path.join(a.legal_dir, "bridge_navhard")
        bdst = os.path.join(a.stage, "vmaxoff_legal", "bridge_navhard")
        os.makedirs(bdst, exist_ok=True)
        for n in ("seam_R7_VMAXOFF.npz", "seam_R7_VMAXOFF.manifest.json"):
            shutil.copyfile(os.path.join(bsrc, n), os.path.join(bdst, n))
        rb = os.path.join(SUITE, "raw", "milestones", "step50400", "bridge_navhard")
        os.makedirs(os.path.join(a.stage, "bridge_navhard"), exist_ok=True)
        for arm in ("R7_A1", "R7_A1_s1"):
            if staged.get(arm):
                for n in (f"seam_{arm}.npz", f"seam_{arm}.manifest.json"):
                    shutil.copyfile(os.path.join(rb, n), os.path.join(a.stage, "bridge_navhard", n))
    if not staged.get("R7_VMAXOFF"):
        print(json.dumps({"status": "UNAVAILABLE", "staged": staged}))
        return 2
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([PY, os.path.join(SUITE, "code", "vmaxoff_legal_summary7.py"), "--root", a.stage, "--splits", a.split],
                       capture_output=True, text=True, env=env)
    src = os.path.join(a.stage, "vmaxoff_legal", f"summary_{a.split}_vmaxoff_legal.json")
    if not os.path.exists(src):
        print(json.dumps({"status": "PARSER_FAILED", "rc": r.returncode, "stderr": r.stderr[-800:]}))
        return 3
    doc = json.load(open(src, encoding="utf-8"))
    doc["backend"] = {"model_arms": "thor (staged: " + json.dumps(staged) + ")",
                      "floors": "dev box (parser's banked floors) => VMAXOFF-vs-floor pairs are MIXED-BACKEND",
                      "policy": "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-04-navsim-thor-backend/RULING_BACKEND_POLICY.md",
                      "parser": "code/vmaxoff_legal_summary7.py run UNCHANGED on a staged scratch root"}
    os.makedirs(a.dest, exist_ok=True)
    out = os.path.join(a.dest, f"summary_{a.split}_vmaxoff_legal.THOR.json")
    json.dump(doc, open(out, "w", encoding="utf-8"), indent=1, default=str)
    open(os.path.join(a.dest, f"summary_{a.split}_vmaxoff_legal.THOR.stdout.txt"), "w", encoding="utf-8").write(r.stdout)
    print(r.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
