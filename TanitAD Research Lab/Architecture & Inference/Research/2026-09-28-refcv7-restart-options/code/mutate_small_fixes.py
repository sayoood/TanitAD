#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deliberate-regression arms for the getstate fix and the I3 fix: each re-introduces HALF a fix
(the plausible partial defect) into a scratch tree that carries the full fix, runs the test file,
and restores the bytes (asserted by git blob). An arm that stays green is a blind spot.

  G1  __getstate__ carries the key but __setstate__ never restores it
  G2  an OLD pickle (no key) is read as True
  I1  the AGENT-head as-trained rule removed (box stamp still read)
  I2  the BOX-head stamp ignored (agent rule still applied)

Usage: mutate_small_fixes.py <tree with both fixes> <log dir>
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ARMS = {
    "G1_setstate_forgets_the_key": (
        "stack/tanitad/data/v2_dataset.py", "tests/test_v2_cache_pickle_state.py",
        '        self.newest_frame_only = bool(s.get("newest_frame_only", False))\n', ""),
    "G2_old_pickle_reads_True": (
        "stack/tanitad/data/v2_dataset.py", "tests/test_v2_cache_pickle_state.py",
        'bool(s.get("newest_frame_only", False))', 'bool(s.get("newest_frame_only", True))'),
    "I1_agent_rule_removed": (
        "stack/tanitad/eval/refcv6_loader.py", "tests/test_refcv6_loader_stamped_queries.py",
        '    _aq = getattr(tr, "agent_queries_as_trained", None)\n', "    _aq = None\n"),
    "I2_box_stamp_ignored": (
        "stack/tanitad/eval/refcv6_loader.py", "tests/test_refcv6_loader_stamped_queries.py",
        '        _pkw = {} if _nq is None else {"n_queries": int(_nq)}\n', "        _pkw = {}\n"),
}


def blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(b) + b).hexdigest()


def main() -> int:
    tree, logs = Path(sys.argv[1]), Path(sys.argv[2])
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2",
               PYTHONPATH=str(tree / "stack"), HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8")
    out = {}
    for arm, (rel, test, old, new) in ARMS.items():
        p = tree / rel
        orig = p.read_bytes()
        txt = orig.decode("utf-8")
        if txt.count(old) != 1:
            out[arm] = {"error": f"anchor count {txt.count(old)}"}
            continue
        p.write_bytes(txt.replace(old, new).encode("utf-8"))
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test],
                               cwd=str(tree / "stack"), env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
        finally:
            p.write_bytes(orig)
        assert blob(p.read_bytes()) == blob(orig)
        (logs / f"pytest_{arm}.log").write_text(r.stdout + r.stderr, encoding="utf-8")
        tail = [ln for ln in r.stdout.splitlines() if "passed" in ln or "failed" in ln][-1:]
        failed = [ln.split("::")[1].split(" ")[0] for ln in r.stdout.splitlines()
                  if ln.startswith("FAILED")]
        out[arm] = {"file": rel, "red": r.returncode != 0, "tail": tail, "failed": failed}
        print(arm, "RED" if r.returncode else "GREEN (BLIND SPOT)", tail, flush=True)
    (logs / "mutations_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0 if all(v.get("red") for v in out.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
