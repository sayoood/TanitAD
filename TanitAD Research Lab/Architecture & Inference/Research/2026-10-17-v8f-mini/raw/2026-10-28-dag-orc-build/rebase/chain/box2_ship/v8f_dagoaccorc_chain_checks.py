#!/usr/bin/env python3
"""PREREG_V8F_DAG_ORC Amendment 1 (re-based on DAG-OACC) on box 2: the DECISIONS the chain makes on files, in one place so a test can run both sides of each (the chain calls this file; nothing here touches box 2).
Derived from v8f_dagorc_chain_checks.py (DAG-based); the run-record clauses for the swap / the bank / w_acc are DAG-OACC's r2 chain's inline record check.

    run-record --run-dir D --train-line F --pin-line F --arm dagoaccorc|dagoaccorcrep --seed S --banked-init-sha M2B_SHA --bank-sha BANK_SHA --w-acc 49.6969 [--steps 3000] [--expect expect.json]
        the TRAINED run's record (K-GATE / K-INIT): box 2's pin == the planned training argv == argv.json; the three argv_sha16 agree; done; steps; arm M2b; the trainer's seed; the run's own selfcheck is a PASS;
        levers.init_ckpt.sha256 == M2b_s<S>'s checkpoint (DAG-OACC's init); the BUILT perturbation == the registered one (A: p 0.3 + the five law flags; REP: p 0 and the defaults); the BUILT swap == DAG-OACC's (dagger_p 0.1,
        DAG-OACC's bank sha256); the BUILT w_acc == W; the composition flag built True on A and absent on REP.   exit 0 PASS | 1 (the failures are printed)
    reached --summary summary.json --arm ARM
        K-REACH (the gate does not cover the loop's draw): A: swap dagger_n / dagger_eligible in 0.1 +- 0.02 AND ego_perturb_n / ego_perturb_eligible (non-swapped eligible) in 0.3 +- 0.02 AND ego_dagger_overlap_n EXACTLY 0
        AND ego_perturb_with_dagger True.  REP: swap in 0.1 +- 0.02 AND ego_perturb_n EXACTLY 0 (eligible > 0) AND the flag absent.   exit 0 PASS | 1 FAIL
    one-lever --ref-argv-json DAG-OACC's argv.json --line F --arm ARM --seed S
        prints ONE-LEVER | NOT-ONE-LEVER: the planned TRAINING line == DAG-OACC's banked argv + exactly the registered change (A: --ego-perturb-p 0.3, the five law flags after it, and --ego-perturb-with-dagger at the end;
        REP: --seed), only --out differing
    pin-sha FILE / argv-sha LINEFILE      sha256 of a file / launch_gate.argv_sha256 of the training argv of a printed line
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("gate_dagoaccorc_assert", os.path.join(HERE, "gate_dagoaccorc_assert.py"))
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

REACH_TOL = 0.02


def argv_sha256(argv) -> str:
    return G.argv_sha256(argv)


def findk(obj, key):
    """The first value of ``key`` at the top level, else in ``levers`` / ``built_levers`` / ``totals``, else anywhere below (None when absent). The composed run record may carry it in any of them."""
    if not isinstance(obj, dict):
        return None
    if key in obj:
        return obj[key]
    for sub in ("levers", "built_levers", "totals"):
        if isinstance(obj.get(sub), dict) and key in obj[sub]:
            return obj[sub][key]
    for v in obj.values():
        if isinstance(v, dict):
            r = findk(v, key)
            if r is not None:
                return r
    return None


def run_record(a) -> list:
    d = a.run_dir
    seed = int(a.seed)
    arm = a.arm
    p = G.ARMS[arm]
    tseed = 10 + seed if arm == "dagoaccorcrep" else seed
    steps = int(a.steps)
    W = float(a.w_acc)
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
        bad.append(f"summary.json levers.init_ckpt.sha256 {str(ic.get('sha256'))[:16]} != M2b_s{seed}'s checkpoint {a.banked_init_sha[:16]} (K-INIT)")
    if ic.get("applied") is not True or ic.get("n_mismatch_after_load") != 0:
        bad.append(f"summary.json init_ckpt not applied / read back: {ic.get('applied')} / {ic.get('n_mismatch_after_load')}")
    if not str(ic.get("path", "")).replace("\\", "/").endswith(exp["init_tail"].format(seed=seed)):
        bad.append(f"summary.json init_ckpt path {ic.get('path')!r} is not ...{exp['init_tail'].format(seed=seed)}")
    ep = lv.get("ego_perturb") or {}
    want = {"ego_perturb_p": p, **({**G.LAW_VALUES, "ego_perturb_seed": seed} if p > 0.0 else G.LAW_DEFAULT), "kin_mode": "yaw"}
    for k, w in want.items():
        if ep.get(k) != w:
            bad.append(f"summary.json built ego_perturb {k} {ep.get(k)!r} != the registered {w!r}")
    wd = findk(sm, "ego_perturb_with_dagger")
    if p > 0.0 and wd is not True:
        bad.append(f"summary.json built ego_perturb_with_dagger {wd!r} on A (must be True)")
    if p == 0.0 and wd not in (False, None):
        bad.append(f"summary.json built ego_perturb_with_dagger {wd!r} on REP (the flag must be absent)")
    if lv.get("dagger_p") != G.SWAP_P:
        bad.append(f"summary.json built dagger_p {lv.get('dagger_p')!r} != {G.SWAP_P}")
    if lv.get("dagger_bank_sha256") != a.bank_sha.lower():
        bad.append(f"summary.json built dagger_bank_sha256 {str(lv.get('dagger_bank_sha256'))[:12]} != DAG-OACC's bank {a.bank_sha[:12]}")
    if lv.get("w_acc") != W:
        bad.append(f"summary.json built w_acc {lv.get('w_acc')!r} != the argv's {W!r} (C-O2)")
    print("DAG-ORC (DAG-OACC base) run record:", "PASS" if not bad else bad, "|", arm, "s%d" % seed, "| trainer seed", tseed, "| argv sha16", sha16, "| steps", sm.get("steps"), "| ego_perturb_p", ep.get("ego_perturb_p"),
          "| w_acc", lv.get("w_acc"))
    return bad


def _count_pair(sm, kn, ke):
    n, e = findk(sm, kn), findk(sm, ke)
    ok = isinstance(n, int) and isinstance(e, int) and not isinstance(n, bool) and not isinstance(e, bool) and e > 0 and 0 <= n <= e
    return n, e, ok


def reached(a) -> int:
    sm = json.load(open(a.summary, encoding="utf-8"))
    p = G.ARMS[a.arm]
    bad = []
    n, e, ok = _count_pair(sm, "dagger_n", "dagger_eligible")
    if not ok:
        bad.append(f"dagger_n {n!r} / dagger_eligible {e!r}: not a count pair (0 <= n <= eligible, eligible > 0)")
        sr = None
    else:
        sr = n / e
        if not (G.SWAP_P - REACH_TOL - 1e-12 <= sr <= G.SWAP_P + REACH_TOL + 1e-12):
            bad.append(f"swapped fraction {n}/{e} = {sr:.4f} outside {G.SWAP_P} +- {REACH_TOL}")
    en, ee, eok = _count_pair(sm, "ego_perturb_n", "ego_perturb_eligible")
    ov, wd = findk(sm, "ego_dagger_overlap_n"), findk(sm, "ego_perturb_with_dagger")
    er = None
    if p == 0.0:
        if not (eok and en == 0):
            bad.append(f"REP: ego_perturb_n {en!r} / eligible {ee!r} (n must be EXACTLY 0, eligible > 0)")
        if wd not in (False, None):
            bad.append(f"REP: ego_perturb_with_dagger {wd!r} (the flag must be absent)")
        if ov not in (0, None):
            bad.append(f"REP: ego_dagger_overlap_n {ov!r} (must be 0 or absent)")
    else:
        if not eok:
            bad.append(f"A: ego_perturb_n {en!r} / ego_perturb_eligible {ee!r}: not a count pair")
        else:
            er = en / ee
            if not (p - REACH_TOL - 1e-12 <= er <= p + REACH_TOL + 1e-12):
                bad.append(f"A: perturbed fraction {en}/{ee} = {er:.4f} outside {p} +- {REACH_TOL}")
        if not (isinstance(ov, int) and not isinstance(ov, bool) and ov == 0):
            bad.append(f"A: ego_dagger_overlap_n {ov!r} must be EXACTLY 0 (no sample is both swapped and perturbed)")
        if wd is not True:
            bad.append(f"A: ego_perturb_with_dagger {wd!r} must be True")
    print("DAG-ORC (DAG-OACC base) K-REACH:", "PASS" if not bad else "FAIL", f"| {a.arm} | swap {n}/{e}" + (f" = {sr:.4f}" if sr is not None else "") + f" (must lie in {G.SWAP_P} +- {REACH_TOL}) | ego {en}/{ee}"
          + (f" = {er:.4f}" if er is not None else "") + f" (A: {p} +- {REACH_TOL}; REP: exactly 0) | overlap {ov!r} (A: exactly 0) | flag {wd!r}" + ("" if not bad else " | " + "; ".join(bad)))
    return 0 if not bad else 1


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
    for k in ("run-dir", "train-line", "pin-line", "arm", "seed", "banked-init-sha", "bank-sha", "w-acc"):
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
