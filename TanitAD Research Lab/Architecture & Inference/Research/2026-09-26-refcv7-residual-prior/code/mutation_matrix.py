#!/usr/bin/env python3
"""refcv7 NEW-1 -- does each test go RED on the defect it names? (the mutation matrix)

Each mutation re-introduces ONE plausible defect into a COPY of the candidate tree's file,
runs ``stack/tests/test_residual_prior.py``, records which tests failed, and restores the
file byte for byte. A mutation that fails NOTHING is a guard that guards nothing.

usage: python mutation_matrix.py <tree>     (the tree is modified and restored in place;
                                            point it at a scratch COPY, never at a worktree)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

KP = "stack/tanitad/models/kinematic_prior.py"
REFC = "stack/tanitad/refs/refc.py"
TRAIN = "stack/scripts/refc_v3_train.py"

MUTATIONS = [
    ("M1 ha0_ext reads the steer at t0-1 (the `ha` index) instead of t0", KP,
     "        steer0 = actions[:, n - 1, 0]\n", "        steer0 = actions[:, n - 2, 0]\n"),
    ("M2 the prior is squeezed through the vocabulary's kappa cap", KP,
     "    k = kappa0.reshape(b, 1, 1).to(torch.float32) + dk\n",
     "    k = (kappa0.reshape(b, 1, 1).to(torch.float32) + dk).clamp(-kappa_cap, kappa_cap)\n"),
    # M3 is an EQUIVALENT mutant, kept on record: `_roll_state` reads the PRIOR's own speed, so
    # dropping `_sample`'s `v = prior[2]` alone changes nothing (MEASURED 2026-09-26: 0 of 27
    # tests fail, and none should). M3b breaks the property itself and must go RED.
    ("M3 (equivalent) drop `_sample`'s `v = prior[2]` only", REFC,
     "            v = prior[2]\n", "            pass\n"),
    ("M3b the sampled fan is rolled at the RAW v0 (refcv6's speed), not the prior's", REFC,
     [("            v = prior[2]\n", "            pass\n"),
      ("        a0, k0, pv = prior\n        return _kp.roll_plan(\n            z, a0, k0, pv,",
       "        a0, k0, pv = prior\n        return _kp.roll_plan(\n            z, a0, k0, v,")]),
    ("M4 NEW-1's keys dropped from RefCModel.DECODER_PASSTHROUGH (the A16 whitelist class)", REFC,
     "        \"residual_prior_ctrl\", \"residual_prior_v\", \"residual_prior_path\")\n",
     "        )\n"),
    ("M5 the prior's speed ignores ego-dropout", REFC,
     "        if ego_keep is not None:\n            v = torch.where(ego_keep.reshape(-1),\n"
     "                            v, self._withheld_ref_speed(v, withheld_speed))\n        return v\n",
     "        return v\n"),
    ("M6 the prior is NOT withheld on dropped rows", REFC,
     "        a0, k0 = _kp.withhold(a0, k0, ego_keep)\n", "        pass\n"),
    ("M7 the prior path is read one tick late", KP,
     "    return torch.tensor([int(h) - 1 for h in horizons], device=device,\n",
     "    return torch.tensor([int(h) for h in horizons if int(h) < 60] + [59], device=device,\n"),
    ("M8 the trainer's F3 cascade re-rolls without the prior", TRAIN,
     "                    prior=_rp_cas)\n", "                    )\n"),
    ("M9 an OFF build computes (and ignores) a prior anyway", REFC,
     "        if self.residual_prior == _kp.RESIDUAL_PRIOR_OFF:\n            if residual_prior is not None:\n",
     "        _kp.withhold(torch.zeros(1), torch.zeros(1), None)\n"
     "        if self.residual_prior == _kp.RESIDUAL_PRIOR_OFF:\n            if residual_prior is not None:\n"),
]


def run(tree: str) -> dict:
    env = dict(os.environ, PYTHONPATH=os.path.join(tree, "stack").replace("\\", "/"),
               OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"))
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "tests/test_residual_prior.py", "-rf"], cwd=os.path.join(tree, "stack"),
                       capture_output=True, text=True, env=env, timeout=1800)
    txt = p.stdout + p.stderr
    failed = sorted(set(re.findall(r"FAILED tests/test_residual_prior.py::(\S+)", txt)))
    m = re.search(r"(\d+) passed", txt)
    f = re.search(r"(\d+) failed", txt)
    return {"passed": int(m.group(1)) if m else 0, "failed": int(f.group(1)) if f else 0,
            "failed_tests": failed, "exit": p.returncode}


def main() -> int:
    tree = os.path.abspath(sys.argv[1])
    only = sys.argv[2:]                       # optional name prefixes, e.g. M3b
    res = {"baseline": run(tree)}
    for m in MUTATIONS:
        name, rel = m[0], m[1]
        if only and not any(name.startswith(o) for o in only):
            continue
        edits = m[2] if len(m) == 3 else [(m[2], m[3])]
        path = os.path.join(tree, rel)
        raw = open(path, "rb").read()
        crlf = raw.count(b"\r\n") == raw.count(b"\n") and raw.count(b"\n") > 0
        text = raw.decode("utf-8").replace("\r\n", "\n")
        missing = [o[:50] for o, _n in edits if text.count(o) != 1]
        if missing:
            res[name] = {"error": f"anchor not found exactly once: {missing}"}
            print(name, "-> ANCHOR ERROR", missing, flush=True)
            continue
        mut = text
        for o, n_ in edits:
            mut = mut.replace(o, n_)
        with open(path, "wb") as fh:
            fh.write((mut.replace("\n", "\r\n") if crlf else mut).encode("utf-8"))
        try:
            res[name] = run(tree)
        finally:
            with open(path, "wb") as fh:
                fh.write(raw)
        assert open(path, "rb").read() == raw
        print(name, "->", res[name]["failed"], "failed", res[name]["failed_tests"][:6], flush=True)
    res["restored_baseline"] = run(tree)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
