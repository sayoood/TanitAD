#!/usr/bin/env python3
"""Mutation self-test for `stop_candidate_probe.py`'s gates: each deliberate regression must turn EXACTLY its gate
RED, and the unmutated dump must pass every gate and report a number.

Runs the probe's own `analyze()` (`--analyze-only`, GPU-free) on mutated copies of a banked dump:
  none              no mutation                               -> every gate green, ZZSTOPPROBE_OK / _SMOKE
  props+1e-3        one proposal coordinate moved 1 mm         -> C-a red (proposals no longer reproduced)
  s64+1e-3          one logit of the 64-set moved 1e-3         -> C-b red (logits no longer reproduced)
  k64_shift         one token's 64-set pick moved by one index -> C-b red (pick no longer reproduced)
  rule_v2           the planner's recorded rule set to v2      -> C-b red (not the shipped v1 aggregate)
  ext65+1e-3        one score_extra logit moved 1e-3           -> C-c red (not the model's own route)
  stop_fed_nonzero  the STOP tensor fed to the scorer != 0     -> C-e red (not the STOP arm's plan)
  stop_csv+1e-6     one token of the STOP csv moved 1e-6       -> C-f red (the re-score no longer matches); runs only on
                    the complete dump once C-f's harness run is banked (it re-uses that PASSed run, no new harness run)

    python eval/selftest_stop_candidate_probe.py [--dump <stop_candidate_dump.npz>]
Prints ZZSELFTEST_OK or ZZSELFTEST_FAIL <case>; exit 0 / 1.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "stop_candidate_probe.py")
DUMP = "D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep015/stop_candidate_dump.npz"


def m_none(d):
    pass


def m_props(d):
    d["props"][0, 0, 0, 0] += 1e-3


def m_s64(d):
    d["s64"][0, 0, 0] += 1e-3


def m_k64(d):
    d["k64"][0] = (d["k64"][0] + 1) % 64


def m_rule(d):
    m = json.loads(str(d["meta"]))
    m["planner_rule"] = "v2_shape"
    d["meta"] = np.array(json.dumps(m))


def m_ext(d):
    d["ext65"][0, 0, 0] += 1e-3


def m_stop(d):
    d["stop_fed"][0, 0] = 0.5


CASES = (("none", m_none, None),
         ("props+1e-3", m_props, "C_a_proposals_reproduced"),
         ("s64+1e-3", m_s64, "C_b_logits_and_pick_reproduced"),
         ("k64_shift", m_k64, "C_b_logits_and_pick_reproduced"),
         ("rule_v2", m_rule, "C_b_logits_and_pick_reproduced"),
         ("ext65+1e-3", m_ext, "C_c_score_extra_route"),
         ("stop_fed_nonzero", m_stop, "C_e_stop_form"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=DUMP)
    a = ap.parse_args()
    base = dict(np.load(a.dump, allow_pickle=False))
    tmp = tempfile.mkdtemp(prefix="selftest_stop_probe_")
    bad = []
    try:
        for name, mut, expect in CASES:
            d = {k: np.array(v, copy=True) for k, v in base.items()}
            mut(d)
            dp, op = os.path.join(tmp, f"{name}.npz"), os.path.join(tmp, f"{name}.json")
            np.savez(dp, **d)
            r = subprocess.run([sys.executable, PROBE, "--analyze-only", "--dump", dp, "--out", op, "--boot", "200",
                                "--no-families"], capture_output=True, text=True)
            last = [ln for ln in r.stdout.splitlines() if ln.startswith("ZZSTOPPROBE")]
            c = json.load(open(op, encoding="utf-8")).get("controls", {}) if os.path.exists(op) else {}
            red = [k for k, v in c.items() if isinstance(v, dict) and v.get("pass") is False]
            if expect is None:
                ok = red == [] and bool(last) and last[-1].split()[0] in ("ZZSTOPPROBE_OK", "ZZSTOPPROBE_SMOKE")
            else:
                ok = red == [expect] and bool(last) and last[-1].startswith("ZZSTOPPROBE_FAIL")
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:18s} red={red} expected={expect} -> "
                  f"{last[-1] if last else 'no marker (rc ' + str(r.returncode) + ')'}", flush=True)
            if not ok:
                bad.append(name)
        # C-f's arm: the banked re-score against a STOP csv with one token moved by 1e-6 must go red
        rescored = ("D:/Projects/TanitAD/data/refe_navtest/score/refe_sub200_ep015_stopzeros/"
                    "refe_sub200_ep015_stopzeros.csv")
        if len(base["token"]) == 200 and os.path.exists(rescored):
            import csv as _csv
            src = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
                   "STOP_navtest/STOP_navtest.csv")
            rows = list(_csv.reader(open(src, encoding="utf-8", newline="")))
            hdr = rows[0]
            it, isc = hdr.index("token"), hdr.index("score")
            want = str(base["token"][0])
            for r in rows[1:]:
                if r[it] == want:
                    r[isc] = repr(float(r[isc]) + 1e-6)
            mp = os.path.join(tmp, "stop_mut.csv")
            with open(mp, "w", encoding="utf-8", newline="") as f:
                _csv.writer(f).writerows(rows)
            op = os.path.join(tmp, "stop_csv_mut.json")
            r = subprocess.run([sys.executable, PROBE, "--analyze-only", "--dump", a.dump, "--out", op, "--boot", "200",
                                "--no-families", "--rescore-stop", "--stop-csv", mp], capture_output=True, text=True)
            last = [ln for ln in r.stdout.splitlines() if ln.startswith("ZZSTOPPROBE")]
            c = json.load(open(op, encoding="utf-8")).get("controls", {}) if os.path.exists(op) else {}
            red = [k for k, v in c.items() if isinstance(v, dict) and v.get("pass") is False]
            ok = red == ["C_f_stop_rescored_through_the_refe_path"] and bool(last) and last[-1].startswith(
                "ZZSTOPPROBE_FAIL")
            print(f"  [{'PASS' if ok else 'FAIL'}] {'stop_csv+1e-6':18s} red={red} -> {last[-1] if last else None}")
            if not ok:
                bad.append("stop_csv+1e-6")
        else:
            print("  [SKIP] stop_csv+1e-6       needs the complete dump and C-f's banked re-score csv")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ZZSELFTEST_OK" if not bad else f"ZZSELFTEST_FAIL {' '.join(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
