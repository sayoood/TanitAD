#!/usr/bin/env python3
"""PREREG_V8F_DAG_ORC on box 2: the DECISIONS the chain makes on files, in one place so a test can run both sides of each (the chain calls this file; nothing here touches box 2).

    run-record --run-dir D --train-line F --pin-line F --arm ARM --seed S --banked-init-sha SHA [--steps 3000] [--expect expect.json]
        the TRAINED run's record (PREREG_V8F_DAG_ORC section 6, K-GATE / K-INIT): box 2's pin == the planned training argv == argv.json; the three argv_sha16 agree; done; steps; arm M2b; the trainer's seed; the run's own selfcheck
        is a PASS; levers.init_ckpt.sha256 == DAG_s<S>'s checkpoint; the BUILT perturbation == the registered one (A: p 0.3 + the five law flags; B: p 0.1; REP: p 0 and the defaults); --dagger-p 0 built and no row swapped.
        exit 0 PASS | 1 (the failures are printed)
    reached --summary summary.json --arm ARM
        K-REACH: the perturbation REACHED the training loop (the gate does not cover the loop's `ego.apply`): A / B ego_perturb_n / ego_perturb_eligible lies in p +- 0.02, REP's ego_perturb_n is EXACTLY 0 with eligible > 0.
        exit 0 PASS | 1 FAIL
    one-lever --ref-argv-json DAG2-0's argv.json --line F --arm ARM --seed S
        prints ONE-LEVER | NOT-ONE-LEVER: the planned TRAINING line == DAG2-0's banked argv + exactly the registered change (A / B: --ego-perturb-p and the five law flags; REP: --seed), only --out differing
    pin-sha FILE / argv-sha LINEFILE      sha256 of a file / launch_gate.argv_sha256 of the training argv of a printed line
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("gate_dagorc_assert", os.path.join(HERE, "gate_dagorc_assert.py"))
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

REACH_TOL = 0.02


def argv_sha256(argv) -> str:
    return G.argv_sha256(argv)


def run_record(a) -> list:
    d = a.run_dir
    seed = int(a.seed)
    arm = a.arm
    p = G.ARMS[arm]
    tseed = 10 + seed if arm == "dagorcrep" else seed
    steps = int(a.steps)
    exp = {"init_tail": G.REAL["init_tail"], **(json.load(open(a.expect, encoding="utf-8")) if a.expect else {})}
    J = lambda f: json.load(open(os.path.join(d, f), encoding="utf-8"))      # noqa: E731
    planned, pinned = G.argv_of(open(a.train_line, encoding="utf-8").read()), G.argv_of(open(a.pin_line, encoding="utf-8").read())
    aj, sm, sc = J("argv.json"), J("summary.json"), J("selfcheck.json")
    bad = []
    if pinned != planned:
        bad.append("the pinned training argv (box 2) is not the planned one")
    if aj.get("argv") != planned:
        got = aj.get("argv") or []
        k = next((k for k in range(min(len(got), len(planned))) if got[k] != planned[k]), min(len(got), len(planned)))
        bad.append(f"argv.json is not the planned training argv (first difference at token {k}: {got[k:k + 2]} vs {planned[k:k + 2]}): the token binds an argv that did not run")
    sha16 = hashlib.sha256(" ".join(planned).encode()).hexdigest()[:16]
    for name, v in (("argv.json sha", aj.get("sha")), ("summary.json argv_sha", sm.get("argv_sha")), ("selfcheck.json argv_sha", sc.get("argv_sha"))):
        if v != sha16:
            bad.append(f"{name} {v} != the planned argv's {sha16}")
    if sm.get("done") is not True:
        bad.append("summary.json done is not true")
    if sm.get("steps") != steps:
        bad.append(f"summary.json steps {sm.get('steps')} != {steps}")
    if (sm.get("arm"), sm.get("seed")) != ("M2b", tseed):
        bad.append(f"summary.json arm / seed {(sm.get('arm'), sm.get('seed'))} != ('M2b', {tseed})")
    if sc.get("declared_vs_built") != "PASS":
        bad.append("the run's own selfcheck.json is not a PASS")
    lv = sm.get("levers") or {}
    ic = lv.get("init_ckpt") or {}
    if ic.get("sha256") != a.banked_init_sha:
        bad.append(f"summary.json levers.init_ckpt.sha256 {str(ic.get('sha256'))[:16]} != DAG_s{seed}'s checkpoint {a.banked_init_sha[:16]} (K-INIT)")
    if ic.get("applied") is not True or ic.get("n_mismatch_after_load") != 0:
        bad.append(f"summary.json init_ckpt not applied / read back: {ic.get('applied')} / {ic.get('n_mismatch_after_load')}")
    if not str(ic.get("path", "")).replace("\\", "/").endswith(exp["init_tail"].format(seed=seed)):
        bad.append(f"summary.json init_ckpt path {ic.get('path')!r} is not ...{exp['init_tail'].format(seed=seed)}")
    ep = lv.get("ego_perturb") or {}
    want = {"ego_perturb_p": p, **({**G.LAW_VALUES, "ego_perturb_seed": seed} if p > 0.0 else G.LAW_DEFAULT), "kin_mode": "yaw"}
    for k, w in want.items():
        if ep.get(k) != w:
            bad.append(f"summary.json built ego_perturb {k} {ep.get(k)!r} != the registered {w!r}")
    if lv.get("dagger_p") != 0.0:
        bad.append(f"summary.json built dagger_p {lv.get('dagger_p')!r} != 0.0")
    if sm.get("dagger_n") != 0:
        bad.append(f"summary.json dagger_n {sm.get('dagger_n')!r} != 0: a row was swapped on a DAG-ORC arm")
    print("DAG-ORC run record:", "PASS" if not bad else bad, "|", arm, "s%d" % seed, "| trainer seed", tseed, "| argv sha16", sha16, "| steps", sm.get("steps"), "| ego_perturb_p", ep.get("ego_perturb_p"))
    return bad


def reached(a) -> int:
    sm = json.load(open(a.summary, encoding="utf-8"))
    p = G.ARMS[a.arm]
    n, e = sm.get("ego_perturb_n"), sm.get("ego_perturb_eligible")
    if p == 0.0:
        ok = isinstance(n, int) and isinstance(e, int) and e > 0 and n == 0 and not isinstance(n, bool)
        print("DAG-ORC perturbation-reached:", "PASS" if ok else "FAIL", f"| {a.arm} | ego_perturb_n {n} (must be exactly 0) | eligible {e} (must be > 0)")
    else:
        ok = isinstance(n, int) and isinstance(e, int) and not isinstance(n, bool) and e > 0 and 0 <= n <= e and p - REACH_TOL - 1e-12 <= n / e <= p + REACH_TOL + 1e-12
        print("DAG-ORC perturbation-reached:", "PASS" if ok else "FAIL", f"| {a.arm} | ego_perturb_n {n} / eligible {e} = {n / e if ok else 'n/a'} (must lie in {p} +- {REACH_TOL})")
    return 0 if ok else 1


def one_lever(a) -> int:
    ref = json.load(open(a.ref_argv_json, encoding="utf-8"))["argv"]
    line = G.argv_of(open(a.line, encoding="utf-8").read())
    try:
        exp = G.expected_from_ref(ref, a.arm, int(a.seed))
    except ValueError:
        print("NOT-ONE-LEVER")
        return 1
    ok = G.drop(line) == exp
    print("ONE-LEVER" if ok else "NOT-ONE-LEVER")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run-record")
    for k in ("run-dir", "train-line", "pin-line", "arm", "seed", "banked-init-sha"):
        r.add_argument("--" + k, required=True)
    r.add_argument("--steps", default="3000")
    r.add_argument("--expect", default="")
    q = sub.add_parser("reached")
    q.add_argument("--summary", required=True)
    q.add_argument("--arm", required=True, choices=sorted(G.ARMS))
    o = sub.add_parser("one-lever")
    o.add_argument("--ref-argv-json", required=True)
    o.add_argument("--line", required=True)
    o.add_argument("--arm", required=True, choices=sorted(G.ARMS))
    o.add_argument("--seed", required=True)
    sub.add_parser("pin-sha").add_argument("file")
    sub.add_parser("argv-sha").add_argument("line_file")
    a = ap.parse_args(argv)
    if a.cmd == "run-record":
        a.run_dir = a.run_dir
        return 1 if run_record(a) else 0
    if a.cmd == "reached":
        return reached(a)
    if a.cmd == "one-lever":
        return one_lever(a)
    if a.cmd == "pin-sha":
        print(hashlib.sha256(open(a.file, "rb").read()).hexdigest())
        return 0
    if a.cmd == "argv-sha":
        print(argv_sha256(G.argv_of(open(a.line_file, encoding="utf-8").read())))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
