#!/usr/bin/env python3
"""Self-test for eval/amendment4_readout.py -- every expectation is a LITERAL, never an expression over the code.

    python eval/selftest_amendment4.py [--metrics <local copy of the live run's metrics.jsonl>]

Prints ZZA4_SELFTEST PASS <n checks> or ZZA4_SELFTEST FAIL <what>. With --metrics it also checks the live run's own bank
lines (the snapshot after epoch 12 must be refused, the one after epoch 13 must be the primary test) and runs the whole
CLI end to end on a scratch copy of real E-6 files relabelled as a fixture.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import amendment4_readout as A  # noqa: E402

checks, fails = 0, []


def check(name, got, want):
    global checks
    checks += 1
    if got != want:
        fails.append(f"{name}: got {got!r}, want {want!r}")


def bank(mode, sets, frames=0):
    return {"event": "bank", "scorer_mode": mode, "onpolicy_sets": sets, "scorer_frames": frames}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=None)
    a = ap.parse_args()

    # 1. the three outcomes and both strict boundaries (the bar is 0.25 and must be EXCEEDED / UNDERCUT)
    check("success", A.outcome(0.30, 0.50), "SUCCESS")
    check("failure", A.outcome(0.10, 0.24), "FAILURE")
    check("straddle", A.outcome(0.20, 0.30), "UNDETERMINED")
    check("lower at the bar", A.outcome(0.25, 0.40), "UNDETERMINED")
    check("upper at the bar", A.outcome(0.00, 0.25), "UNDETERMINED")
    check("ep012 numbers", A.outcome(-0.033, 0.198), "FAILURE")

    # 2. eligibility: the snapshot INTO epoch N trained epoch N-1, which must be on-policy on EVERY bank line
    banks = {10: [bank(None, None, 120264)], 11: [bank(None, None, 120264), bank("onpolicy", 0)],
             12: [bank("onpolicy", 1618)], 13: [bank("onpolicy", 10231), bank("onpolicy", 10231)]}
    check("ep012 refused", A.eligibility(12, banks)[0], False)
    check("ep013 eligible", A.eligibility(13, banks)[:2], (True, 1))
    check("ep014 second (restart inside epoch 13 stays on-policy)", A.eligibility(14, banks)[:2], (True, 2))
    # MUTATION GUARD: an epoch that ENDED on-policy (with sets) but STARTED fixed is not a whole on-policy epoch -- a
    # reader of the LAST bank line would admit it
    mut = {11: [bank(None, None, 120264), bank("onpolicy", 5)], 12: [bank("onpolicy", 1618)]}
    check("started-fixed epoch refused", A.eligibility(12, mut)[0], False)
    check("empty on-policy bank refused", A.eligibility(13, {12: [bank("onpolicy", 0)]})[0], False)
    check("missing epoch refused", A.eligibility(20, banks)[0], False)
    # names: a snapshot INTO epoch NNN, or the final model one past the last epoch with a bank line
    check("name ep013", A.snapshot_epoch("sub200_ep013", banks), 13)
    check("name final", A.snapshot_epoch("sub200_final", banks), 14)
    check("name neither", A.snapshot_epoch("sub200_foo", banks), None)
    check("final eligible, second on-policy point", A.eligibility(A.snapshot_epoch("sub200_final", banks), banks)[:2], (True, 2))
    check("final with no banks", A.snapshot_epoch("sub200_final", {}), None)
    # after a DECISION later reads are reported, not gating; only an UNDETERMINED hands the decision on
    td = tempfile.mkdtemp(prefix="a4dec_")
    try:
        def put(name, o):
            os.makedirs(os.path.join(td, "proptable", name), exist_ok=True)
            json.dump({"primary": {"outcome": o}}, open(os.path.join(td, "proptable", name, "amendment4.json"), "w"))
        check("decided: none earlier (rank 1 is never asked)", A.decided_before(banks, 1, td), None)
        check("decided: earlier file missing", A.decided_before(banks, 2, td), {"name": "sub200_ep013", "outcome": "MISSING"})
        put("sub200_ep013", "UNDETERMINED")
        check("decided: earlier UNDETERMINED hands it on", A.decided_before(banks, 2, td), None)
        put("sub200_ep013", "FAILURE")
        check("decided: earlier FAILURE decides", A.decided_before(banks, 2, td), {"name": "sub200_ep013", "outcome": "FAILURE"})
    finally:
        shutil.rmtree(td, ignore_errors=True)

    # 3. the training-side pool: set-weighted means over the epoch's steps only, literal arithmetic
    op = {3725: {"sets": 9, "pick": 0.9, "random": 0.1, "best": 0.9},      # before the epoch: excluded
          3726: {"sets": 1, "pick": 0.4, "random": 0.3, "best": 0.7},
          3727: {"sets": 3, "pick": 0.6, "random": 0.4, "best": 0.8},
          4129: {"sets": 5, "pick": 0.0, "random": 0.5, "best": 0.9}}      # the next epoch: excluded
    t = A.training_skill(op, 3726, 4129)
    check("pool sets", t["sets"], 4)
    check("pool pick", t["pick"], 0.55)
    check("pool random", t["random"], 0.375)
    check("pool best", t["best"], 0.775)
    check("pool skill", t["skill"], 0.4375)
    check("empty window", A.training_skill(op, 5000, 6000), None)

    # 4. the live run's own lines, and the whole CLI on a relabelled fixture
    if a.metrics:
        b, starts, opl = A.read_metrics(a.metrics)
        check("live: ep012 refused", A.eligibility(12, b)[0], False)
        check("live: ep013 is the primary test", A.eligibility(13, b)[:2], (True, 1))
        check("live: epoch 12 starts at step 3726", starts.get(12), 3726)
        tmp = tempfile.mkdtemp(prefix="a4fx_")
        try:
            for nm in ("sub200_ep011", "sub200_ep012"):
                os.makedirs(os.path.join(tmp, "proptable", nm))
                shutil.copy(os.path.join(A.DATA, "proptable", nm, "readout.json"), os.path.join(tmp, "proptable", nm))
            fx = "sub200_ep013"          # the ep012 files wearing the ep013 name: exercises every read and write
            os.makedirs(os.path.join(tmp, "proptable", fx))
            os.makedirs(os.path.join(tmp, "points"))
            shutil.copy(os.path.join(A.DATA, "proptable", "sub200_ep012", "readout.json"),
                        os.path.join(tmp, "proptable", fx))
            shutil.copy(os.path.join(A.DATA, "points", "sub200_ep012.json"), os.path.join(tmp, "points", fx + ".json"))
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            # ⛔ every fixture document goes to the TEMP dir: the fixture names are REAL snapshot names, and on
            # 2026-09-26 a fixture run overwrote the real RESULT_A4_sub200_ep013.md in eval/ (restored from the index)
            p = subprocess.run([sys.executable, os.path.join(HERE, "amendment4_readout.py"), "--name", fx,
                                "--metrics", a.metrics, "--data", tmp, "--doc-dir", tmp],
                               capture_output=True, text=True, env=env)
            check("cli exit", p.returncode, 0)
            check("cli outcome (ep012's interval)", "ZZA4 sub200_ep013 FAILURE" in p.stdout, True)
            j = json.load(open(os.path.join(tmp, "proptable", fx, "amendment4.json"), encoding="utf-8"))
            check("cli rank", j["rank_among_onpolicy_snapshots"], 1)
            check("cli dac above 0.60", j["secondary"]["dac_within_scene_auc"]["above_0p60"], False)
            check("cli has the pick-vs-STOP pair", j["secondary"]["pick_vs_stop"]["separated"], True)
            check("cli wrote its doc into the temp dir", os.path.exists(os.path.join(tmp, f"RESULT_A4_{fx}.md")), True)
            p2 = subprocess.run([sys.executable, os.path.join(HERE, "amendment4_readout.py"), "--name", "sub200_ep012",
                                 "--metrics", a.metrics, "--data", tmp, "--doc-dir", tmp],
                                capture_output=True, text=True, env=env)
            check("cli refuses ep012", (p2.returncode, "NOT_ELIGIBLE" in p2.stdout), (2, True))
            if A.eligibility(14, b)[0]:
                # the fixture's ep013 read just decided FAILURE: an ep014 read must say it is not gating
                fx2 = "sub200_ep014"
                os.makedirs(os.path.join(tmp, "proptable", fx2))
                shutil.copy(os.path.join(A.DATA, "proptable", "sub200_ep012", "readout.json"),
                            os.path.join(tmp, "proptable", fx2))
                shutil.copy(os.path.join(A.DATA, "points", "sub200_ep012.json"), os.path.join(tmp, "points", fx2 + ".json"))
                p3 = subprocess.run([sys.executable, os.path.join(HERE, "amendment4_readout.py"), "--name", fx2,
                                     "--metrics", a.metrics, "--data", tmp, "--doc-dir", tmp],
                                    capture_output=True, text=True, env=env)
                check("cli ep014 after a FAILURE is not gating",
                      (p3.returncode, "gating=no (decided by sub200_ep013: FAILURE)" in p3.stdout), (0, True))
                j2 = json.load(open(os.path.join(tmp, "proptable", fx2, "amendment4.json"), encoding="utf-8"))
                check("cli ep014 decides", j2["primary"]["decides"], False)
            else:
                print("SKIP ep014 check: the live metrics have no on-policy epoch-13 bank line yet")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("SKIP live-run checks: no --metrics given")

    if fails:
        print("ZZA4_SELFTEST FAIL " + " | ".join(fails))
        return 1
    print(f"ZZA4_SELFTEST PASS {checks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
