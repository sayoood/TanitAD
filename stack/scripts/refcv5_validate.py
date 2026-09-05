#!/usr/bin/env python3
"""refcv5_validate.py -- run the `V-RC5-READY` tiny-rig validation.

Executes the arms of
`TanitAD Research Lab/Architecture & Inference/Research/
2026-09-05-refcv5-training-readiness/SPEC.md`, which was banked BEFORE any arm
ran, and reports each gate G1-G5 as written.

⛔ The four deliberate regressions (R1-R4) are configurations that would each
TRAIN, CONVERGE and WRITE A CHECKPOINT while a stamped weight supervised
nothing. Two of them are not hypothetical -- they are arms this stream has
already banked. R1 is the one no flag guard can see: the flag is set, the
camera is really built, the term really runs, and its gradient is 8.7e-11.

⚠️ The converse control C1 must NOT be refused. A guard that also blocks the
live projection arm is a capability loss dressed as safety.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STACK = HERE.parent
REPO = STACK.parent
TRAINER = HERE / "refc_v3_train.py"

BASE = ["--arm", "hier", "--smoke", "--device", "cpu", "--synth-episodes", "4"]

ARMS = [
    ("A0", ["--steps", "3"], "reach"),
    ("A0r", ["--steps", "3"], "reach"),
    ("A1", ["--steps", "3", "--agents", "oracle", "--agent-queries", "8"],
     "reach"),
    ("R1", ["--steps", "3", "--image-hw", "256", "640",
            "--agents", "oracle", "--agent-rig-camera", "nominal",
            "--agent-w-ground", "0.5"], "refuse"),
    ("R2", ["--steps", "3", "--agents", "head", "--w-agent", "1.0"], "refuse"),
    ("R3", ["--steps", "3", "--agents", "off", "--w-agent", "1.0"], "refuse"),
    ("R4", ["--steps", "3", "--sampler", "ddim", "--w-u0", "0"], "refuse"),
    ("C1", ["--steps", "3", "--image-hw", "256", "640",
            "--agents", "oracle", "--agent-rig-camera", "nominal",
            "--agent-w-project", "0.2"], "reach"),
]


def run(name, extra, out_root, timeout=3600):
    out = out_root / name
    cmd = [sys.executable, str(TRAINER), "--out", str(out), *BASE, *extra]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="6",
               PYTHONPATH=str(STACK) + os.pathsep + str(REPO / "taniteval"))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=env, errors="replace")
        rc, so, se = r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired:
        rc, so, se = -9, "", "TIMEOUT"
    ck = out / "ckpt.pt"
    summ = out / "summary.json"
    d = {"arm": name, "argv": extra, "rc": rc,
         "reached_ckpt": ck.exists(), "reached_summary": summ.exists(),
         "ckpt_sha256": (hashlib.sha256(ck.read_bytes()).hexdigest()
                         if ck.exists() else None),
         "tail": (se or so)[-700:].replace("\n", " | ")}
    print("[%s] rc=%d ckpt=%s summary=%s" % (name, rc, d["reached_ckpt"],
                                             d["reached_summary"]), flush=True)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    root = Path(a.out or tempfile.mkdtemp(prefix="rc5_val_"))
    root.mkdir(parents=True, exist_ok=True)
    print("== V-RC5-READY ==  arms under %s" % root, flush=True)

    res = {}
    for name, extra, want in ARMS:
        res[name] = run(name, extra, root)
        res[name]["expected"] = want

    # ---- the gates, exactly as the SPEC wrote them ----------------------- #
    gates = {}
    g1 = {n: (not res[n]["reached_ckpt"] and res[n]["rc"] != 0)
          for n in ("R1", "R2", "R3", "R4")}
    gates["G1 every deliberate regression is REFUSED"] = {
        "pass": all(g1.values()), "per_arm": g1,
        "detail": {n: res[n]["tail"][-220:] for n in g1}}
    gates["G2 the converse control C1 is NOT refused"] = {
        "pass": bool(res["C1"]["reached_summary"]),
        "detail": res["C1"]["tail"][-220:]}
    gates["G3 A0/A0r/A1 reach summary.json"] = {
        "pass": all(res[n]["reached_summary"] for n in ("A0", "A0r", "A1")),
        "detail": {n: res[n]["reached_summary"] for n in ("A0", "A0r", "A1")}}
    same = (res["A0"]["ckpt_sha256"] is not None
            and res["A0"]["ckpt_sha256"] == res["A0r"]["ckpt_sha256"])
    gates["G4 the replicate control is byte-identical"] = {
        "pass": bool(same),
        "detail": "A0 %s / A0r %s"
                  % ((res["A0"]["ckpt_sha256"] or "NONE")[:16],
                     (res["A0r"]["ckpt_sha256"] or "NONE")[:16])}

    # G5: the gradient probe, through the shipped preflight function
    sys.path.insert(0, str(HERE))
    import refcv5_preflight as pf
    pf.RESULTS.clear()
    pf.check_gradient_reachability()
    row = pf.RESULTS[-1]
    dead = row.get("dead_terms") or []
    grads = row.get("grads") or {}
    gates["G5 the gradient probe separates DEAD from LIVE"] = {
        "pass": (row["state"] != "INCONCLUSIVE"
                 and dead == ["agent_w_ground"]
                 and grads.get("agent_w_project", 0.0) > 1e-6),
        "detail": row["detail"]}

    ok = all(g["pass"] for g in gates.values())
    print("\n-- gates ------------------------------------------------------")
    for k, v in gates.items():
        print("  [%s] %s" % ("PASS" if v["pass"] else "FAIL", k), flush=True)
    print("\n== V-RC5-READY: %s ==" % ("PASS" if ok else "FAIL"), flush=True)

    payload = {"spec": "V-RC5-READY", "pass": ok, "gates": gates,
               "arms": res, "out_root": str(root)}
    if a.json:
        Path(a.json).write_text(json.dumps(payload, indent=2),
                                encoding="utf-8")
        print("wrote %s" % a.json, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
