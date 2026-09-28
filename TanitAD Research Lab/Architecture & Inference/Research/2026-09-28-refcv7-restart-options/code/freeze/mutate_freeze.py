#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deliberate-regression arms for the freeze test (`test_refcv7_admitted_freeze.py`).

Each arm re-introduces ONE plausible defect into a SCRATCH copy of the frozen tree's files, runs the
test file, and restores the bytes (verified by git blob). An arm that stays green is a blind spot.

  M0  the launch code itself (run separately: raw/freeze/pytest_M0_LAUNCH_code_RED.log)
  M1  the ARGV table loses one row (core.route_head) -- G-DVB would then refuse the build
  M2  a later pass UNDOES the model-side freeze (requires_grad restored after declaring)
  M3  the existence filter removed (a module the build did not construct is demanded)
  M4  the converter's state-drop refusal removed (a live tensor's moments silently discarded)
  M5  the model table drops the tac8 pair (only --no-strategic declared)

Usage: mutate_freeze.py <frozen tree> <log dir>
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PY = sys.executable
ARMS = {
    "M1_argv_table_loses_route_head": (
        "stack/tanitad/train/declared_vs_built.py",
        '    ("core.route_head", "--no-strategic", "the route readout; route_loss_applied False"),\n',
        ""),
    "M2_freeze_undone_after_declaring": (
        "stack/tanitad/refs/refc_v3.py",
        "        declare_grad_unreachable(mod, why)\n        done[path] = why\n",
        "        declare_grad_unreachable(mod, why)\n        mod.requires_grad_(True)\n"
        "        done[path] = why\n"),
    "M3_existence_filter_removed": (
        "stack/tanitad/train/declared_vs_built.py",
        "    want = {p: w for p, w in want.items() if p not in _bp or _built_module(model, p)}\n",
        "    want = dict(want)\n"),
    "M4_converter_state_drop_refusal_removed": (
        "stack/scripts/refcv7_ckpt_freeze_convert.py",
        "    if with_state and not allow_state_drop:\n",
        "    if False:\n"),
    "M5_model_table_drops_the_tac8_pair": (
        "stack/tanitad/refs/refc_v3.py",
        '    ("core.decoder.lat_to_anchor", "graft_tac8_prior",\n',
        '    ("core.decoder.lat_to_anchor__MUTANT", "graft_tac8_prior",\n'),
}


def blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(b) + b).hexdigest()


def main() -> int:
    tree, logs = Path(sys.argv[1]), Path(sys.argv[2])
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2",
               PYTHONPATH=str(tree / "stack"), HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8")
    summary = {}
    for arm, (rel, old, new) in ARMS.items():
        p = tree / rel
        orig = p.read_bytes()
        txt = orig.decode("utf-8")
        if txt.count(old) != 1:
            summary[arm] = {"error": f"anchor count {txt.count(old)}"}
            continue
        p.write_bytes(txt.replace(old, new).encode("utf-8"))
        try:
            r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                "tests/test_refcv7_admitted_freeze.py"], cwd=str(tree / "stack"),
                               env=env, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
        finally:
            p.write_bytes(orig)
        assert blob(p.read_bytes()) == blob(orig)
        (logs / f"pytest_{arm}.log").write_text(r.stdout + r.stderr, encoding="utf-8")
        failed = [ln.split("::")[1].split(" ")[0] for ln in r.stdout.splitlines()
                  if ln.startswith("FAILED")]
        tail = [ln for ln in r.stdout.splitlines() if "passed" in ln or "failed" in ln][-1:]
        summary[arm] = {"file": rel, "rc": r.returncode, "red": r.returncode != 0,
                        "failed": failed, "tail": tail}
        print(arm, "RED" if r.returncode else "GREEN (BLIND SPOT)", tail, flush=True)
    (logs / "mutations_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0 if all(v.get("red") for v in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
