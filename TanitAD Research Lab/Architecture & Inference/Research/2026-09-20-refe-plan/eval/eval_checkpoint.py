"""ONE learning-curve point: a REFe checkpoint on a navtest token set -> seam -> PDMS -> floors + families.

Chains the four pre-registered steps (SPEC_NAVTEST), each in the interpreter it was validated with:
  1. `refe_navtest_seam.py`   (driverl venv)  REFePlanner.infer -> seam + frame control (gated, 1 mm)
  2. `score_navtest_refe.py`  (driverl venv)  W3's harness, guards C1-C3
  3. `parse_navtest6.py`      (TanitAD venv)  W3's CV/STOP/HUMAN floors on the SAME tokens, paired
                                               log-cluster bootstrap (the EvalFlyWheel's script, unchanged)
  4. `families6.py`           (TanitAD venv)  the four families (strategic UNAVAILABLE by design)
The shared scripts label the arm `R6_A1` / `refcv6`; in THIS output that arm is REFe (relabelled
here, computation untouched). Nothing is quoted unless steps 1-2 passed their gates.

  python eval/eval_checkpoint.py --ckpt D:/.../snap_epoch004.pt --name ep004 \
      --tokens D:/.../A1_sub200_tokens.json
Writes D:/Projects/TanitAD/data/refe_navtest/points/<name>.json and prints ZZPOINT_OK <name> <PDMS>.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# ⛔ NOT the old %TEMP% venv: a temp-file cleanup deleted part of it on 2026-09-24 (torch could no
# longer import). Rebuilt offline 2026-09-26 into a permanent path and verified by reproducing a
# banked seam bit-for-bit -- eval/EVAL_VENV.md.
DRIVERL_PY = os.environ.get("REFE_DRIVERL_PY", "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe")
TANITAD_PY = os.environ.get("REFE_TANITAD_PY", "C:/Users/Admin/venvs/tanitad/Scripts/python.exe")
EV6 = ("C:/Users/Admin/ev6/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/"
       "navsim/code")
DATA = "D:/Projects/TanitAD/data/refe_navtest"
DZ = "C:/Users/Admin/dz/DriveZero/DriveRL"
EXPORT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"


def env_driverl():
    e = dict(os.environ)
    e.update({"DZ_ROOT": DZ, "NUPLAN_DATA_ROOT": "D:/Projects/TanitAD/data/nuplan",
              "NUPLAN_MAPS_ROOT": "D:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0",
              "DRIVERL_EVAL_ROOT": DZ, "DRIVERL_EVAL_NUPLAN_ROOT": f"{DZ}/nuplan-devkit",
              "DRIVERL_EVAL_CONFIG_PATH": f"{DZ}/release/configs/driverl_teacher.yaml",
              "DRIVERL_EVAL_CHECKPOINT_PATH": f"{DZ}/release/checkpoints/checkpoint_2400.pt",
              "PYTHONPATH": f"{DZ}/nuplan-devkit;{DZ}/src",
              "REFE_BACKBONE_ROOT": "D:/Projects/TanitAD/data/backbones", "REFE_SIM_HZ": "10",
              "OMP_NUM_THREADS": "4", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
    return e


def run(cmd, cwd, env, log):
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    return rc, open(log, encoding="utf-8", errors="replace").read()


def relabel(obj):
    """The shared scripts name the arm under test `R6_A1` / `refcv6` and extra arms `R6_<name>`;
    here the arm is REFe. KEYS are renamed structurally -- never a text replace, which would also
    rewrite any path or note that happens to contain the substring."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str):
                k = k.replace("R6_A1", "REFe").replace("R6_", "REFe_")
                k = "REFe" if k == "refcv6" else k
            out[k] = relabel(v)
        return out
    if isinstance(obj, list):
        return [relabel(v) for v in obj]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--frames", default=f"{DATA}/frames")
    ap.add_argument("--prev", nargs="*", default=[],
                    help="earlier points to PAIR against, as <name> (their score CSV is found by "
                         "label refe_<name> on the same tokens): the learning curve is read as "
                         "paired deltas, not as independent points")
    a = ap.parse_args()
    t0 = time.time()
    pdir = os.path.join(DATA, "points", a.name)
    os.makedirs(pdir, exist_ok=True)
    seam = os.path.join(DATA, "seams", f"refe_{a.name}.npz")
    out = {"name": a.name, "ckpt": a.ckpt, "tokens": a.tokens, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    # 1 seam
    # SPEC E-6: the same forward pass also writes every proposal + the scorer's logits, so the
    # selection diagnosis (eval/proposal_table.py --reuse-dump) needs no second inference
    rc, txt = run([DRIVERL_PY, "refe_navtest_seam.py", "--ckpt", a.ckpt, "--frames", a.frames,
                   "--tokens", a.tokens, "--out", seam, "--arm", f"REFe_{a.name}",
                   "--dump-proposals", os.path.join(DATA, "proptable", a.name, "proposals.npz")],
                  HERE, env_driverl(), os.path.join(pdir, "1_seam.log"))
    m = re.search(r"ZZSEAM_OK (\d+) ([0-9.]+)", txt)
    if not m:
        out["verdict"] = "SEAM_FAILED"
        json.dump(out, open(os.path.join(DATA, "points", f"{a.name}.json"), "w"), indent=1)
        print(f"ZZPOINT_FAIL {a.name} seam"); return 1
    out["seam"] = json.load(open(os.path.splitext(seam)[0] + ".report.json"))
    # 2 score
    label = f"refe_{a.name}"
    rc, txt = run([DRIVERL_PY, "score_navtest_refe.py", "--label", label, "--seam", seam,
                   "--tokens", a.tokens, "--out", f"{DATA}/score"],
                  HERE, dict(os.environ, PYTHONIOENCODING="utf-8"), os.path.join(pdir, "2_score.log"))
    rep = None
    for line in reversed(txt.splitlines()):
        if line.startswith("{") and '"status"' in line:
            rep = json.loads(line); break
    out["score"] = rep
    if not rep or rep.get("status") != "PASS":
        out["verdict"] = "SCORE_FAILED"
        json.dump(out, open(os.path.join(DATA, "points", f"{a.name}.json"), "w"), indent=1)
        print(f"ZZPOINT_FAIL {a.name} score"); return 1
    csv_p = f"{DATA}/score/{label}/{label}.csv"
    # 3 floors + paired intervals (the arm is labelled R6_A1 by the shared script; it is REFe)
    par = os.path.join(pdir, "3_parse.json")
    extra = []
    for p in a.prev:
        pc = f"{DATA}/score/refe_{p}/refe_{p}.csv"
        if os.path.exists(pc):
            extra.append(f"R6_{p}={pc}")
    cmd = [TANITAD_PY, "parse_navtest6.py", "--arm-csv", csv_p, "--tokens", a.tokens,
           "--label", f"REFe-{a.name}", "--out", par]
    if extra:
        cmd += ["--extra"] + extra
    run(cmd, EV6, dict(os.environ, PYTHONIOENCODING="utf-8"), os.path.join(pdir, "3_parse.log"))
    # 4 families
    fam = os.path.join(pdir, "4_families.json")
    run([TANITAD_PY, "families6.py", "--seam", seam, "--inputs", EXPORT, "--stage", "1",
         "--label", f"REFe-{a.name}", "--out", fam, "--n-boot", "2000"],
        EV6, dict(os.environ, PYTHONIOENCODING="utf-8"), os.path.join(pdir, "4_families.log"))
    for k, p in (("floors", par), ("families", fam)):
        if os.path.exists(p):
            out[k] = relabel(json.load(open(p, encoding="utf-8")))
        else:
            out[k] = None                         # absent, and said so -- never silently dropped
    out["seconds"] = round(time.time() - t0, 1)
    out["verdict"] = "OK"
    json.dump(out, open(os.path.join(DATA, "points", f"{a.name}.json"), "w"), indent=1)
    print(f"ZZPOINT_OK {a.name} PDMS={rep['summary_x100_4dp']['PDMS']} n={rep['log_successful']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
