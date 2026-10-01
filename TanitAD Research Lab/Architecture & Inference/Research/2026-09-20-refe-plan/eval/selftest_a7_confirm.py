#!/usr/bin/env python3
"""Mutation self-test for `a7_confirm.py`'s validity gates: each deliberate regression must turn EXACTLY its gate RED
(and the statistic must then NOT be read), and the unmutated artifacts must pass every gate. GPU-free
(`analyze --no-families --no-docs`, outputs to a temp dir; the registered artifacts are never written).

  none                no mutation                                         -> gates green, ZZA7_OK
  repaired_x+1mm      one repaired pose's x moved 1 mm                      -> (a) red
  repaired_h35        the repaired t = 3.5 s heading moved                  -> (a) red
  repaired_h40_other  the repaired t = 4.0 s heading != native heading[18]  -> (a) red
  native_x+1mm        the native record moved (seam != to_navsim(native))   -> (b) red
  native_pick         one recorded pick changed                             -> (b) red
  harness_fail        the repaired run's status set to FAIL                 -> (c) red

    python eval/selftest_a7_confirm.py
Prints ZZSELFTEST_OK or ZZSELFTEST_FAIL <case>; exit 0 / 1.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a7_confirm as A7            # noqa: E402

GATE = {"a": "a_only_the_4s_heading_differs", "b": "b_shipped_seam_reproduces_the_pipeline",
        "c": "c_every_harness_run_PASS_every_token_valid"}


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="selftest_a7_")
    R0 = dict(np.load(A7.SEAM_REPAIRED))
    N0 = dict(np.load(A7.NATIVE, allow_pickle=False))
    logdir0 = os.path.join(A7.WD, "logs")

    def rep_mut(fn):
        def f():
            r = {k: np.array(v, copy=True) for k, v in R0.items()}
            fn(r)
            p = os.path.join(tmp, "rep.npz")
            np.savez(p, **r)
            return ["--repaired-seam", p]
        return f

    def nat_mut(fn):
        def f():
            nn = {k: np.array(v, copy=True) for k, v in N0.items()}
            fn(nn)
            p = os.path.join(tmp, "nat.npz")
            np.savez(p, **nn)
            return ["--native", p]
        return f

    def log_fail():
        d = os.path.join(tmp, "logs")
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(logdir0, d)
        p = os.path.join(d, "score_repaired.log")
        txt = open(p, encoding="utf-8", errors="replace").read().replace('"status": "PASS"', '"status": "FAIL"')
        open(p, "w", encoding="utf-8").write(txt)
        return ["--logdir", d]

    def m_x(r):
        r["poses"][0, 3, 0] += 1e-3

    def m_h35(r):
        r["poses"][0, 6, 2] += 1e-3

    def m_h40(r):
        r["poses"][0, 7, 2] += 1e-3

    def m_nx(nn):
        nn["native"][0, 4, 0] += 1e-3

    def m_pick(nn):
        nn["pick"][0] = (nn["pick"][0] + 1) % 64

    cases = (("none", lambda: [], None), ("repaired_x+1mm", rep_mut(m_x), "a"), ("repaired_h35", rep_mut(m_h35), "a"),
             ("repaired_h40_other", rep_mut(m_h40), "a"), ("native_x+1mm", nat_mut(m_nx), "b"),
             ("native_pick", nat_mut(m_pick), "b"), ("harness_fail", log_fail, "c"))
    bad = []
    try:
        for name, prep, expect in cases:
            extra = prep()
            op = os.path.join(tmp, f"{name}.json")
            r = subprocess.run([sys.executable, os.path.join(HERE, "a7_confirm.py"), "analyze", "--no-families",
                                "--no-docs", "--boot", "200", "--out-json", op] + extra,
                               capture_output=True, text=True, cwd=HERE)
            last = [ln for ln in r.stdout.splitlines() if ln.startswith("ZZA7")]
            g = json.load(open(op, encoding="utf-8")).get("gates", {}) if os.path.exists(op) else {}
            red = sorted(k for k, v in GATE.items() if isinstance(g.get(v), dict) and g[v].get("pass") is False)
            read = os.path.exists(op) and "statistic" in json.load(open(op, encoding="utf-8"))
            if expect is None:
                ok = red == [] and bool(last) and last[-1].startswith("ZZA7_OK") and read
            else:
                ok = red == [expect] and bool(last) and last[-1].startswith("ZZA7_FAIL") and not read
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:19s} red={red} expected={expect} statistic_read={read} -> "
                  f"{last[-1] if last else 'no marker (rc ' + str(r.returncode) + ')'}", flush=True)
            if not ok:
                bad.append(name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ZZSELFTEST_OK" if not bad else f"ZZSELFTEST_FAIL {' '.join(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
