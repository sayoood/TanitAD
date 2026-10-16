#!/usr/bin/env python3
"""Tests for S1c: the S1 aux drivable-area head trained for a longer BUDGET (raw/2026-10-16-s1c-build/S1C_DESIGN.md).  CPU only, no pod, no GPU, no checkpoint.
Run:  PYTHONIOENCODING=utf-8 python refe/test_s1c.py   ->   ZZTEST_PASS <n>      (needs bash on PATH; a check that cannot run prints ZZTEST_SKIP and is NOT counted)

WHAT IS PROVEN, AND HOW (CLAUDE.md "built, tested, and unreachable from its caller"):
  1. ONE TABLE.  launch_gate.AUXS1C_BUDGETS, s1aux_checks.S1C_BUDGETS and the argv the CHAIN builds (arm_args, extracted verbatim from s1aux_pod.sh and run under bash) agree, and each budget
     fits inside ONE epoch of the audited 174,912 training sets (so `--epochs` at its default of 1.0 does not truncate it).
  2. THE GATE.  The chain's own argv for each S1c arm is judged by the REAL profile (refe_aux_s1c): CLEAN; and every budget flag edited, dropped (a budget must be NAMED), crossed with the other
     budget, the seed, the head levers, the arm class and the arm name is REFUSED.  S1's nine arms are still CLEAN under refe_aux_s1 (the profile was generalised, not changed).
  3. G-DVB reads the schedule.  `max_updates` is only `consumed` in the registry (never observed), so an --epochs that is too small stops a run SILENTLY early (TL-1).  The S1c profile re-derives the
     schedule from the preflight's data event; a smaller built total FAILS, and so does the deliberate-regression arm `schedule_truncated`.
  4. REACHABILITY.  The REAL scorer_finetune.main() (the real AuxDrivable head, 2,440,450 parameters, against a 256-wide stub trunk and a synthetic label store) is run at the SHAPE of each budget
     (updates x accum, evals at the eval tags): the requested update count is reached (start, done, last train row, the head optimiser's own Adam step counter), the schedule ENDS there (lr 0.0),
     every head tensor moved, G12 reads a non-zero gradient at the declared weight AND exactly 0.0 at weight 0.  Each lever is then BROKEN in the source and the same reach check must go RED:
     `--max-updates` not honoured, `--epochs` too small, the schedule not ending at N, the head never stepped, the aux loss dropped, the head cut from the loss.
  5. s1aux_checks.run-check accepts a finished S1c arm at ITS update count and eval tags, and refuses every deviation (wrong count, a missing tag, a missing dump, a schedule that did not end).
  6. THE NAVTEST PIN (backlog item: s1aux_pod.sh pinned the CRLF form).  The chain and G-SUITE's E10 identify the list by its CR-stripped bytes: LF and CRLF forms both pass, a different list fails.
  7. THE CHAIN, END TO END (the dev box only: it needs the real aux label store, so it is SKIPPED, not counted, on the pod).  The REAL s1aux_pod.sh in test_s1aux_chain's sandbox, with the
     stub trainer made budget-aware: S1AUX_ARMS="H1u10k-s0" runs ONLY that arm (one preflight, one gate mint under refe_aux_s1c, one trainer start, the registered argv token for token),
     proceeds with the logged WARN when RL-S1 never logged ZZRLS1_DONE, banks the S1c evidence under arms/H1u10k-s0/, and a trainer that ends early (the TL-1 epoch cap) is REFUSED by the run-check.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKGROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PKGROOT / "eval"))
import gate_checks as C  # noqa: E402
import launch_gate as L  # noqa: E402
import s1aux_checks as SC  # noqa: E402
import test_s1aux_chain as TC  # noqa: E402  (bash helpers: header_text / fn_body / arm_args_of ...)
import test_launch_gate as TLG  # noqa: E402  (preflight fixtures)

CHAIN_TEXT = (HERE / "s1aux_pod.sh").read_text(encoding="utf-8")
N_OK = 0
SKIPS: list = []
EVID: dict = {}                 # S1C_EVIDENCE_OUT=<file.json>: the numbers the reachability proof read (written at the end of main)
TRAIN_SETS = SC.TRAIN_SETS


def ok(cond, msg=""):
    global N_OK
    assert cond, msg
    N_OK += 1


def skip(why):
    SKIPS.append(why)
    print("ZZTEST_SKIP", why)


def tmpdir():
    return tempfile.mkdtemp(prefix="s1c_", dir=os.environ.get("TMP") or None)


# ============================================================================================ 1. one table
def tables():
    B = L.AUXS1C_BUDGETS
    ok(set(L.AUXS1C_ARM_CLASS) == {f"H1{b}-s0" for b in B} and set(L.AUXS1C_ARM_CLASS.values()) == {f"H1{b}" for b in B}, L.AUXS1C_ARM_CLASS)
    ok(all(L.AUXS1C_ARM_SEED[a] == 0 for a in L.AUXS1C_ARM_CLASS) and L.AUXS1C_SEEDS == (0,))               # ONE seed: the replicate is a follow-up PI decision
    ok(not (set(L.AUXS1C_ARM_CLASS) & set(L.AUXS1_ARM_CLASS)))                                              # no S1 arm name is reused (the HF repo keeps S1's arms/<ARM>/ untouched)
    for b, bud in B.items():
        s = SC.S1C_BUDGETS[f"H1{b}"]
        ok(s == {"updates": bud["max_updates"], "batch": bud["batch"], "accum": bud["accum"], "eval_every": bud["eval_every"]}, (b, s, bud))
        rec = L.AUXS1C_ARM_RECIPE[f"H1{b}"]
        ok(all(rec[k] == v for k, v in bud.items()) and all(rec[k] == v for k, v in L.AUXS1_RECIPE.items() if k not in bud))      # S1's recipe with ONLY the budget moved
        # the budget fits in ONE epoch of the audited training set: `--epochs` stays at its default 1.0 and still reaches max_updates (scorer_finetune.py:1363-1365)
        reach = int(math.ceil(TRAIN_SETS / bud["batch"] * 1.0 / bud["accum"]))
        ok(reach >= bud["max_updates"], (b, reach))
        ok(bud["max_updates"] % bud["eval_every"] == 0 and bud["max_updates"] % 25 == 0)                     # the last update is an eval AND a logged train row (the end-of-schedule check reads it)
        ok(bud["max_updates"] * bud["batch"] * bud["accum"] <= TRAIN_SETS)                                  # no row is seen twice: a fresh-frame regime (S1: 25,600 rows, no repeats)
    P = L.PROFILES["refe_aux_s1c"]
    ok(P["arm_rules"] == "auxs1" and P["runner"] == "scorer" and P["schedule_reach"] is True and P["suite"] == L.AUXS1C_SUITE and P["mintable"] and P["require_commit"])
    ok(P["required_flags"]["--max-updates"] is None and L.PROFILES["refe_aux_s1"]["required_flags"]["--max-updates"] == "400")      # S1 is unchanged
    ok(P["forbidden_flags"] == L.PROFILES["refe_aux_s1"]["forbidden_flags"] and P["data_flags"] == L.PROFILES["refe_aux_s1"]["data_flags"] and P["checks"] == L.PROFILES["refe_aux_s1"]["checks"])
    ok("schedule_reach" not in L.PROFILES["refe_aux_s1"] and "arm_recipe" not in L.PROFILES["refe_aux_s1"])
    for e in L.AUXS1C_SUITE:
        ok((HERE / e["script"]).is_file() and e["token"] in (HERE / e["script"]).read_text(encoding="utf-8"), e)
    ok(L.AUXS1C_SUITE[-1]["script"] == "test_s1c.py" and L.AUXS1C_SUITE[:-1] == L.AUXS1_SUITE)
    ok(L.AUXS1C_SUITE[-1].get("args") == ["--no-e2e"])                                             # the pod run DECLARES the e2e scope (a skip would fail G-SUITE)
    ok(C.DVB_KINDS_SCORER["max_updates"] == "consumed")                                                     # the registry kind is unchanged: the schedule is observed by the profile's extra rows


# ============================================================================================ 2. the chain's argv through the real gate
def chain_argv(tmp):
    P = TC.resolved_vars(CHAIN_TEXT, ("U400", "TARGETS", "IMAGES", "ONP", "HO", "HOLOGS", "OUT", "D", "LABELS", "CALIB"))
    out = {}
    for arm in list(L.AUXS1C_ARM_CLASS) + ["H1-s0", "N-s0", "H2-s0"]:
        rc, o, err = TC.arm_args_of(CHAIN_TEXT, arm, tmp)
        ok(rc == 0, (arm, err))
        out[arm] = o
    for bad in ("H1u10k-s1", "H1u3k-s0", "H1u10k", "X"):
        ok(TC.arm_args_of(CHAIN_TEXT, bad, tmp)[0] != 0, bad)                                               # an unregistered S1c arm has no argv
    return P, out


def gate(tmp):
    P, argv_of = chain_argv(tmp)
    prof = lambda arm, name="refe_aux_s1c": {**L.PROFILES[name], "declared_arm": arm}                    # noqa: E731
    toks = lambda s: L.parse_args_string(s)                                                              # noqa: E731

    def refs(arm, argv, name="refe_aux_s1c"):
        return L.profile_refusals(prof(arm, name), argv)

    def clean(arm, argv, name="refe_aux_s1c"):
        r = refs(arm, argv, name)
        ok(r == [], (arm, r))

    def red(arm, argv, frag, name="refe_aux_s1c"):
        r = refs(arm, argv, name)
        ok(any(frag in x for x in r), (frag, arm, r))
    # ---- the chain's argv, character for character, against the registered budgets
    for arm, cls in L.AUXS1C_ARM_CLASS.items():
        bud = L.AUXS1C_BUDGETS[cls[2:]]
        a = toks(argv_of[arm])
        for flag, key in (("--max-updates", "max_updates"), ("--batch", "batch"), ("--accum", "accum"), ("--eval-every", "eval_every")):
            ok(a.count(flag) == 1 and a[a.index(flag) + 1] == str(bud[key]), (arm, flag))
        clean(arm, a)
        ok("--aux-drv" in a and "--aux-drv-feed" not in a and "--w-aux-drv" in a and "--aux-drv-lr" in a and "--dump-scores" in a)       # an H1 argv: head on, feed OFF
        h1 = toks(argv_of["H1-s0"])
        ok(a[a.index("--ckpt"):a.index("--require-pdm") + 1] == h1[h1.index("--ckpt"):h1.index("--require-pdm") + 1])               # everything up to the budget is H1-s0's, token for token
    # the DIFF of H1u10k-s0 against H1-s0 is exactly the budget, the out dir and the dump dir (the lever is the ONLY thing that moved)
    d10, d1 = toks(argv_of["H1u10k-s0"]), toks(argv_of["H1-s0"])
    flat = lambda a: [(a[i], a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else None) for i in range(len(a)) if a[i].startswith("--")]      # noqa: E731
    moved = {f for f in dict(flat(d10)) if dict(flat(d10))[f] != dict(flat(d1)).get(f)}
    ok(moved == {"--max-updates", "--batch", "--accum", "--eval-every", "--dump-scores", "--out"}, moved)    # --batch 8 / --accum 1 are NAMED (H1-s0 leaves them to the parser's 8 / 8)
    ok(dict(flat(d10))["--batch"] == "8" and "--batch" not in d1)
    # S1's nine arms are still clean under refe_aux_s1 (generalised, not changed), and refe_aux_s1c does not accept them
    for arm in ("N-s0", "H1-s0", "H2-s0"):
        clean(arm, toks(argv_of[arm]), "refe_aux_s1")
        ok(any("unknown arm" in x for x in refs(arm, toks(argv_of[arm]))))
    for arm in L.AUXS1C_ARM_CLASS:
        ok(any("unknown arm" in x for x in refs(arm, toks(argv_of[arm]), "refe_aux_s1")))                  # an S1c arm name is not an S1 arm
    # ---- the real gate REFUSES every deviation
    a10, a2 = toks(argv_of["H1u10k-s0"]), toks(argv_of["H1u2k-s0"])
    for arm, a in (("H1u10k-s0", a10), ("H1u2k-s0", a2)):
        bud = L.AUXS1C_BUDGETS[arm[2:-3]]
        for flag, bad in (("--max-updates", str(bud["max_updates"] + 1)), ("--max-updates", "400"), ("--accum", "3"), ("--batch", "4"), ("--eval-every", "400"), ("--lr", "3e-5"), ("--warmup", "10"),
                          ("--aux-drv-lr", "1e-4"), ("--w-aux-drv", "0.2"), ("--tau-s", "0.2"), ("--lam", "0.3")):
            red(arm, C._set(a, flag, bad), "requires " + flag)
        for flag, nv in (("--max-updates", 1), ("--batch", 1), ("--accum", 1), ("--eval-every", 1)):
            red(arm, C._drop(a, flag, nv), flag)                                                             # a budget flag left out: required (max-updates) or NOT NAMED (the others)
        for flag in ("--batch", "--accum", "--eval-every"):
            red(arm, C._drop(a, flag, 1), "to be NAMED")
        red(arm, C._set(a, "--seed", "1"), "registered with --seed 0")
        red(arm, a + ["--aux-drv-feed", "on"], "H1 class")
        red(arm, a + ["--epo"], "FORBIDS")
        red(arm, a + ["--aux-drv-shuffle-seed", "7"], "FORBIDS")
        red(arm, C._drop(a, "--aux-drv", 0), "requires --aux-drv")
    red("H1u10k-s0", a2, "requires --max-updates 10000")                                                     # the other budget's argv under this arm
    red("H1u2k-s0", a10, "requires --max-updates 2000")
    red("H1u10k-s0", a2, "requires --accum 1")
    red("H1u2k-s0", a10, "requires --accum 8")
    red("H1u10k-s0", a2, "requires --eval-every 5000")
    red("H1u10k-s0", toks(argv_of["H1-s0"]), "requires --max-updates 10000")                                 # S1's own H1 argv is not an S1c arm
    red("H1u10k-s0", toks(argv_of["H2-s0"]), "H1 class")
    red("H1u10k-s0", toks(argv_of["N-s0"]), "H1 class")
    red(None, a10, "must be NAMED (--arm-name)")                                                             # undeclared: an argv alone cannot say which budget it tests
    clean("H1u10k-s0", a10)
    # ---- the CLI (what the chain calls): the arm name belongs to its profile
    def cli(*args):
        return subprocess.run([sys.executable, str(HERE / "launch_gate.py"), *args], capture_output=True, text=True, encoding="utf-8")
    rc = cli("refusals", "--profile", "refe_aux_s1c", "--arm-name", "H1u10k-s0", "--args", argv_of["H1u10k-s0"])
    ok(rc.returncode == 0 and "ZZGATE-CLEANZZ" in rc.stdout, rc.stdout[-400:] + rc.stderr[-400:])
    rc = cli("refusals", "--profile", "refe_aux_s1c", "--arm-name", "H1u10k-s0", "--args", argv_of["H1u2k-s0"])
    ok(rc.returncode == 5 and "ZZGATE-REFUSEDZZ" in rc.stdout and "requires --max-updates 10000" in rc.stdout, rc.stdout[-400:])
    rc = cli("refusals", "--profile", "refe_aux_s1c", "--arm-name", "H1-s0", "--args", argv_of["H1-s0"])
    ok(rc.returncode != 0 and "is not an arm of refe_aux_s1c" in (rc.stdout + rc.stderr), rc.stderr[-300:])
    rc = cli("refusals", "--profile", "refe_aux_s1", "--arm-name", "H1u10k-s0", "--args", argv_of["H1u10k-s0"])
    ok(rc.returncode != 0 and "is not an arm of refe_aux_s1" in (rc.stdout + rc.stderr), rc.stderr[-300:])
    with tempfile.TemporaryDirectory() as td_:
        rc = cli("run", "--profile", "refe_aux_s1c", "--args", argv_of["H1u10k-s0"], "--pkg", td_, "--out-dir", str(Path(td_) / "o"))
        ok(rc.returncode != 0 and "mints for ONE named arm" in (rc.stdout + rc.stderr), rc.stderr[-300:])
    return P, argv_of


# ============================================================================================ 3. G-DVB reads the schedule
def dvb(argv_of):
    import scorer_finetune as S
    for arm, cls in L.AUXS1C_ARM_CLASS.items():
        a = L.parse_args_string(argv_of[arm])
        prof = {**L.PROFILES["refe_aux_s1c"], "declared_arm": arm}
        ns, err = C.parse_scorer_argv(S.build_parser(), a)
        ok(err is None, err)

        def pf(train_sets, argv=a):
            rows = TLG.aux_pf_rows(argv, data_mut={"train_sets": train_sets})
            return C.load_scorer_preflight(TLG.rls1_write(Path(tempfile.mkdtemp(dir=os.environ.get("TMP") or None)) / "p.jsonl", rows))
        ev = C.check_dvb_scorer(a, pf(TRAIN_SETS), prof)
        ok(ev["status"] == "PASS", ev["reasons"])
        rows = {r["dest"]: r for r in ev["details"]["rows"]}
        ok(rows["max_updates"]["declared"] == ns.max_updates == rows["max_updates"]["built"] and ev["details"]["observed_rows"] == 20, ev["details"]["observed_rows"])     # S1's 19 rows + the schedule row
        # the deliberate-regression arm: a quarter of the training sets cannot reach the budget
        ev = C.check_dvb_scorer(a, pf(TRAIN_SETS), prof, arm="schedule_truncated")
        ok(ev["status"] == "FAIL" and "--max-updates: declared" in " ".join(ev["reasons"]) and ev["arm"] == "schedule_truncated", ev["reasons"])
        # and so does a real --epochs that is too small (named: the trainer would stop at ceil(N_sets / batch x epochs / accum) < max_updates)
        short = a + ["--epochs", "0.2"]
        ev = C.check_dvb_scorer(short, pf(TRAIN_SETS, short), prof)
        built = int(math.ceil(TRAIN_SETS / ns.batch * 0.2 / ns.accum))
        ok(built < ns.max_updates and ev["status"] == "FAIL" and f"built {built}" in " ".join(ev["reasons"]), (built, ev["reasons"]))
        # fewer training sets than the audited ones (a carve gone wrong) cannot reach it either
        few = int(ns.max_updates * ns.batch * ns.accum) - ns.batch * ns.accum
        ev = C.check_dvb_scorer(a, pf(few), prof)
        ok(ev["status"] == "FAIL", ev["reasons"])
        ev = C.check_dvb_scorer(a, pf(int(ns.max_updates * ns.batch * ns.accum)), prof)                    # exactly enough rows: reaches, to the update
        ok(ev["status"] == "PASS", ev["reasons"])
        # the S1 profile does not carry the rows (its 19 stay 19): the check is the S1c profile's, not a change of S1
        s1 = TLG.aux_pf_rows(L.parse_args_string(argv_of["H1-s0"]))
        pf1 = C.load_scorer_preflight(TLG.rls1_write(Path(tempfile.mkdtemp(dir=os.environ.get("TMP") or None)) / "p1.jsonl", s1))
        ev = C.check_dvb_scorer(L.parse_args_string(argv_of["H1-s0"]), pf1, {**L.PROFILES["refe_aux_s1"], "declared_arm": "H1-s0"})
        ok(ev["status"] == "PASS" and ev["details"]["observed_rows"] == 19, ev["reasons"])


# ============================================================================================ 4. reachability: the REAL main() at the shape of each budget
def reach_problems(res, n, accum, eval_every, *, batch=2):
    """What a run of budget (n updates, accum, eval_every) must show.  [] = the budget is REACHED and the head TRAINED.  The same function judges every broken variant."""
    import torch
    bad = []
    rows = res["rows"]["sft.jsonl"]
    by = {}
    for r in rows:
        by.setdefault(r.get("event"), []).append(r)
    st, dn = by.get("start", []), by.get("done", [])
    if res.get("rc_pre") != 0 or res.get("rc_run") != 0:
        bad.append(f"exit codes {res.get('rc_pre')} {res.get('rc_run')}")
        return bad
    if len(st) != 1 or st[0].get("updates") != n or st[0].get("micro_per_update") != accum or st[0].get("batch") != batch:
        bad.append(f"start event {[(s.get('updates'), s.get('micro_per_update'), s.get('batch')) for s in st]} != ({n}, {accum}, {batch})")
    if len(dn) != 1 or dn[0].get("updates") != n:
        bad.append(f"done event {[d.get('updates') for d in dn]} != {n}")
    tr = [r for r in by.get("train", [])]
    if not tr or tr[-1].get("update") != n or tr[-1].get("of") != n:
        bad.append(f"last train row {(tr[-1].get('update'), tr[-1].get('of')) if tr else None} != ({n}, {n})")
    elif abs(tr[-1].get("lr", 1.0)) > 1e-12 or abs(tr[-1].get("aux_lr", 1.0)) > 0.01 * 2e-4:
        # the scorer's `lr` is lr_at(N) = 0.0 exactly; the head's `aux_lr` is read off its optimiser, i.e. the rate of the LAST STEP, lr_at(N - 1) x ratio (S1's H1-s0: 4.0e-9 of a 2e-4 peak)
        bad.append(f"the schedule did not END at update {n}: lr {tr[-1].get('lr')} aux_lr {tr[-1].get('aux_lr')}")
    tags = {r.get("tag") for r in by.get("eval", [])}
    want = SC.eval_tags({"updates": n, "eval_every": eval_every})
    if set(want) - tags:
        bad.append(f"eval tags {sorted(map(str, tags))} lack {sorted(set(want) - tags)}")
    atags = {r.get("tag") for r in by.get("aux_eval", [])}
    if set(want) - atags:
        bad.append(f"aux_eval tags lack {sorted(set(want) - atags)}")
    aux = res.get("aux")
    if aux is None:
        bad.append("aux_drv.pt missing")
        return bad
    if aux.get("update") != n:
        bad.append(f"aux_drv.pt saved at update {aux.get('update')} != {n}")
    # the head's own optimiser stepped exactly n times on every parameter (Adam allocates state lazily: a parameter that never received a gradient has NO entry)
    steps = {int(v["step"]) for v in aux["opt"]["state"].values()} if aux["opt"]["state"] else set()
    n_par = len(aux["opt"]["param_groups"][0]["params"])
    if steps != {n} or len(aux["opt"]["state"]) != n_par:
        bad.append(f"head Adam: steps {sorted(steps)} on {len(aux['opt']['state'])} of {n_par} parameters, want {{{n}}} on all")
    # every head tensor MOVED from its own initialisation (the head draws from its own RNG stream: rebuilt with the same config and seed it is bit-identical to the run's step-0 head)
    import aux_drivable_head as AUX
    cfg = AUX.AuxDrivableConfig(**aux["meta"]["config"])
    init = AUX.AuxDrivable(cfg).head.state_dict()
    still = [k for k, v in aux["head"].items() if k in init and torch.equal(v, init[k])]
    if still:
        bad.append(f"{len(still)} head tensors never moved: {still[:4]}")
    return bad


def mutate(src, a, b):
    assert src.count(a) == 1, (src.count(a), a)
    return src.replace(a, b)


def reachability():
    import test_aux_s1_wiring as TW
    src = (HERE / "scorer_finetune.py").read_text(encoding="utf-8")
    H1 = TW.H1_ARGS
    # the stub bank has 100 training sets: batch 2 x accum 1 reaches 50 updates in ONE epoch (the epoch cap, scorer_finetune.py:1363); batch 2 x accum 2 needs --epochs 2.0 to reach 50.  The logged train rows are every 25th update.
    shapes = (("u10k", 50, 1, 2, 25, ["--batch", "2", "--accum", "1", "--max-updates", "50", "--eval-every", "25"]),             # the shape of H1u10k: accum 1, a mid eval and the final one
              ("u2k", 50, 2, 2, 25, ["--batch", "2", "--accum", "2", "--max-updates", "50", "--eval-every", "25", "--epochs", "2.0"]))     # the shape of H1u2k: accum > 1 (2 stub epochs: the real arm's 128,000 rows are 0.73 of ONE epoch)
    pre_cache = {}
    for name, n, accum, batch, ev, extra in shapes:
        td = tmpdir()
        res = TW.smoke_aux(src, td, tag=f"s1c_{name}", extra=H1 + extra, max_updates=n, accum=accum)
        bad = reach_problems(res, n, accum, ev, batch=batch)
        ok(bad == [], (name, bad))
        # G12 at the declared weight: every head sub-module has a finite, non-zero gradient; the zero control (weight 0.0) reads EXACTLY 0.0 on every one
        g = TW.gate(res)
        ok(g is not None and g["ok"] and g["per_group_declared"] and all(math.isfinite(v) and v > 0.0 for v in g["per_group_declared"].values()), g and g.get("per_group_declared"))
        ok(g["per_group_w0"] and set(g["per_group_w0"]) == set(g["per_group_declared"]) and all(v == 0.0 for v in g["per_group_w0"].values()), g["per_group_w0"])
        ok(g["adam_head_params"] > 0 and g["adam_head_params_with_state"] == g["adam_head_params"])
        ok(res["pass_json"] is not None)
        ab = [r["aux_bce"] for r in res["rows"]["sft.jsonl"] if r.get("event") == "train"]
        ok(ab and all(math.isfinite(x) for x in ab))
        pre_cache[name] = Path(td) / "out" / "PREFLIGHT_PASS.json"
        tr = [r for r in res["rows"]["sft.jsonl"] if r.get("event") == "train"]
        st = [r for r in res["rows"]["sft.jsonl"] if r.get("event") == "start"][0]
        dn = [r for r in res["rows"]["sft.jsonl"] if r.get("event") == "done"][0]
        steps = sorted({int(v["step"]) for v in res["aux"]["opt"]["state"].values()})
        EVID.setdefault("positive_runs", {})[name] = {
            "requested": {"max_updates": n, "batch": batch, "accum": accum, "eval_every": ev}, "start_event": {k: st.get(k) for k in ("updates", "micro_per_update", "batch", "train_sets")},
            "done_updates": dn.get("updates"), "last_train_row": {k: tr[-1].get(k) for k in ("update", "of", "lr", "aux_lr", "aux_bce")}, "first_train_row_aux_bce": tr[0].get("aux_bce"),
            "head_adam_steps": steps, "head_tensors": len(res["aux"]["head"]), "head_tensors_with_adam_state": len(res["aux"]["opt"]["state"]),
            "eval_tags": [r.get("tag") for r in res["rows"]["sft.jsonl"] if r.get("event") == "eval"], "aux_eval_tags": [r.get("tag") for r in res["rows"]["sft.jsonl"] if r.get("event") == "aux_eval"],
            "G12_per_group_grad_abs_declared": g["per_group_declared"], "G12_per_group_grad_abs_weight0_control": g["per_group_w0"], "G12_adam_head_params_with_state": [g["adam_head_params_with_state"], g["adam_head_params"]]}

    def broken(name, n, accum, ev, extra, new_src=None, env=None, preflight=False, shape=None):
        """a run of the (mutated) trainer that reuses the unmutated preflight's PASS file (the mutation is in the RUN path only) -> reach problems"""
        td = tmpdir()
        os.makedirs(os.path.join(td, "out"), exist_ok=True)
        if not preflight:
            shutil.copy(pre_cache[shape or "u10k"], os.path.join(td, "out", "PREFLIGHT_PASS.json"))
        res = TW.smoke_aux(new_src or src, td, tag=f"s1c_bad_{name}", extra=H1 + extra, max_updates=n, accum=accum, preflight=preflight, env=env)
        if not preflight and res.get("rc_pre") is None:
            res["rc_pre"] = 0
        return reach_problems(res, n, accum, ev), res
    # ---- the same reach check must go RED when the lever is broken IN THE SOURCE (each snippet occurs exactly once)
    x = ["--batch", "2", "--accum", "1", "--max-updates", "25", "--eval-every", "25"]
    bad, _ = broken("ctrl", 25, 1, 25, x)
    EVID["mutants"] = {"control_unmutated": bad}
    ok(bad == [], bad)                                                                                       # control: the same request, unmutated, is clean
    bad, _ = broken("maxupd", 25, 1, 25, x, mutate(src, "total = min(total, a.max_updates)", "pass"))
    EVID["mutants"]["max_updates_not_honoured"] = bad
    ok(any("done event" in b or "start event" in b for b in bad), bad)                                       # --max-updates not honoured: the run ends at the epoch cap (50), not 25
    bad, _ = broken("epochs", 25, 1, 25, x + ["--epochs", "0.2"])
    EVID["mutants"]["epochs_too_small_TL1"] = bad
    ok(any("start event" in b for b in bad) and any("done event" in b for b in bad), bad)                    # TL-1: --epochs 0.2 ends the run at 10 of 25, silently, rc 0
    bad, _ = broken("lr_end", 25, 1, 25, x, mutate(src, "max(total - a.warmup, 1)", "max(2 * total - a.warmup, 1)"))
    EVID["mutants"]["schedule_does_not_end_at_N"] = bad
    ok(any("did not END" in b for b in bad), bad)                                                            # the cosine no longer reaches 0 at N
    bad, _ = broken("headstep", 25, 1, 25, x, mutate(src, "optH.step()", "pass"))
    EVID["mutants"]["head_optimiser_never_steps"] = bad
    ok(any("head Adam" in b for b in bad) and any("never moved" in b for b in bad), bad)                     # the head is never stepped: no Adam state, no tensor moved
    # ---- G12 (the preflight) goes RED when the aux loss never reaches backward() or the head is cut from the loss, and withholds PREFLIGHT_PASS.json
    for env_name in ("REFE_AUX_MUTATE_DROP_LOSS", "REFE_AUX_MUTATE_DETACH_HEAD"):
        td = tmpdir()
        res = TW.smoke_aux(src, td, tag=f"s1c_{env_name}", extra=H1 + x, max_updates=25, accum=1, run=False, env={env_name: "1"})
        g = TW.gate(res)
        ok(res["rc_pre"] == 3 and res["pass_json"] is None and g is not None and not g["ok"], (env_name, res["rc_pre"], g and g.get("ok")))
        EVID["mutants"][env_name] = {"preflight_rc": res["rc_pre"], "PREFLIGHT_PASS_written": res["pass_json"] is not None, "G12_ok": g.get("ok"), "G12_per_group_declared": g.get("per_group_declared")}


# ============================================================================================ 5. run-check at the budget's own count
def run_check_budgets():
    ev = {"event": "data", "train_sets": SC.TRAIN_SETS, "heldout_sets": SC.HELDOUT_SETS, "heldout_logs": SC.HELDOUT_LOGS, "train_rows_in_heldout_logs": 0,
          "lane_labels": {"teacher_lane_train": SC.TLANE_MIN_ROWS, "teacher_lane_heldout": 3137, "pdm_train": 174912, "pdm_heldout": 3137}}
    root = Path(tmpdir())

    def mk(arm, *, n=None, drop_tags=(), drop_files=(), last_lr=0.0, last_update=None, batch=8, accum=None, skip_dump_tag=None):
        bud = SC.arm_budget(arm)
        n = bud["updates"] if n is None else n
        accum = bud["accum"] if accum is None else accum
        d = root / f"{arm}_{n}_{'-'.join(drop_tags)}_{'-'.join(drop_files)}_{last_lr}_{last_update}_{batch}_{accum}_{skip_dump_tag}"
        shutil.rmtree(d, ignore_errors=True)
        (d / "dump").mkdir(parents=True)
        tags = SC.eval_tags(bud)
        rows = [ev, {"event": "start", "updates": n, "micro_per_update": accum, "batch": batch}]
        rows += [{"event": "train", "update": u, "of": n, "bceA": 0.5, "lossB": 0.4, "aux_bce": 0.5, "lr": 1e-4} for u in range(25, n, 25 * max(1, n // 100))]
        rows.append({"event": "train", "update": last_update or n, "of": n, "bceA": 0.5, "lossB": 0.4, "aux_bce": 0.4, "lr": last_lr})
        for t in tags:
            if t in drop_tags:
                continue
            rows += [{"event": "eval", "tag": t}, {"event": "aux_eval", "tag": t}]
        rows.append({"event": "done", "updates": n})
        (d / "sft.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        (d / "model_sft_B.pt").write_bytes(b"B" * 2000)
        (d / "aux_drv.pt").write_bytes(b"a")
        for t in tags:
            if t != skip_dump_tag:
                (d / "dump" / f"scores_{t}_B.npz").write_bytes(b"s")
                (d / "dump" / f"head_pred_{t}.npz").write_bytes(b"h")
        for f in drop_files:
            (d / f).unlink()
        return d
    keep = SC.MODEL_B_MIN_BYTES
    SC.MODEL_B_MIN_BYTES = 1000
    try:
        for arm in L.AUXS1C_ARM_CLASS:
            good, why = SC.run_check(mk(arm), arm)
            ok(good, why)
            ok(f"updates {SC.arm_budget(arm)['updates']}" in why)
            bud = SC.arm_budget(arm)
            ok(not SC.run_check(mk(arm, n=400), arm)[0])                                                     # S1's 400 is not this arm's budget
            ok(not SC.run_check(mk(arm, n=bud["updates"] - 25), arm)[0])
            ok(not SC.run_check(mk(arm, drop_tags=("step0",)), arm)[0])
            ok(not SC.run_check(mk(arm, drop_tags=(SC.eval_tags(bud)[1],)), arm)[0])                         # the MID eval is a required tag too
            ok(not SC.run_check(mk(arm, skip_dump_tag=SC.eval_tags(bud)[1]), arm)[0])                        # ... and so are its dumps
            ok(not SC.run_check(mk(arm, last_lr=1e-5), arm)[0])                                             # the schedule did not end at 0
            ok(not SC.run_check(mk(arm, last_update=bud["updates"] - 25), arm)[0])                          # the last logged row is not update N
            ok(not SC.run_check(mk(arm, batch=4), arm)[0])
            ok(not SC.run_check(mk(arm, accum=3), arm)[0])
            ok(not SC.run_check(mk(arm, drop_files=("aux_drv.pt",)), arm)[0])
        ok(SC.eval_tags(SC.arm_budget("H1u10k-s0")) == ["step0", "update5000", "update10000"] and SC.eval_tags(SC.arm_budget("H1u2k-s0")) == ["step0", "update1000", "update2000"])
        ok(SC.eval_tags(SC.arm_budget("H1-s0")) == ["step0", "update400"] and SC.arm_budget("N-s0")["updates"] == 400)             # S1 unchanged
        ok(SC.arm_class("H1u10k-s0") == "H1u10k")
    finally:
        SC.MODEL_B_MIN_BYTES = keep


# ============================================================================================ 6. the navtest pin and the dd size
def navtest_pin(tmp):
    import test_aux_drivable_head as TA
    base = b'{"token_to_log_name": {"a": "L1", "b": "L2"}}\n{"x": 1}\n'
    crlf, lf = base.replace(b"\n", b"\r\n"), base
    other = base.replace(b'"L2"', b'"L3"')
    sc = Path(tmp) / "nt.sh"
    # the chain's own function, extracted verbatim from the script
    one = TC.one_line_fn(CHAIN_TEXT, "navtest_sha")
    ok(one and "tr -d" in one and "sha256sum" in one, one)
    TC.write_sh(sc, one + '\nnavtest_sha "$1"\n')
    res = {}
    for nm, raw in (("lf", lf), ("crlf", crlf), ("other", other)):
        f = Path(tmp) / f"nt_{nm}.json"
        f.write_bytes(raw)
        rc, out, err = TC.bash_run(f'bash "{TC.posix(sc)}" "{TC.posix(f)}"')
        ok(rc == 0 and len(out) == 64, (nm, err))
        res[nm] = out
    ok(res["lf"] == res["crlf"] == hashlib.sha256(lf).hexdigest() and res["other"] != res["lf"])
    rc, out, _ = TC.bash_run(f'bash "{TC.posix(sc)}" "{TC.posix(Path(tmp) / "absent.json")}"')
    ok(out == hashlib.sha256(b"").hexdigest() and out != res["lf"])                                          # an absent file hashes as empty: never the pin
    pin = re.search(r"NAVTEST_SHA=([0-9a-f]{64})", CHAIN_TEXT).group(1)
    ok(pin == "3ba1c96488f157fd2ca7c57c83abe439b4c9680ebdb1eadf65c34ac323d42d44" and list(TA.NAVTEST_LF_SHA.values()) == [pin])        # the chain and E10 pin the same list
    ok(CHAIN_TEXT.count('[ "$(navtest_sha "$NAVTEST_JSON")" = "$NAVTEST_SHA" ] || refuse') == 1 and "0b9c63895d4e8da4205e194c90379cbf986badd1baf8beab948adb3743a33700" not in CHAIN_TEXT)
    # E10's identity rule (test_aux_drivable_head.navtest_identity), on the real record of the store manifest
    crlf_rec = next(iter(TA.NAVTEST_LF_SHA))
    ok(TA.navtest_identity(b"x", hashlib.sha256(b"x").hexdigest())[0])                                      # the raw form still passes
    ok(not TA.navtest_identity(other, crlf_rec)[0] and not TA.navtest_identity(b"", crlf_rec)[0])           # a different list matches neither form
    # the REAL list, when this machine has it (the pod: $AUX_NAVTEST; the dev box: the repo copy, LF or CRLF)
    cands = [os.environ.get("AUX_NAVTEST", "")] + ([str(PKGROOT.parents[3] / "FlyWheels" / "TanitAD_EvalFlyWheel" / "incoming" / "2026-09-19-navsim-estimator-and-route-leak" / "raw" / "cluster_maps" / "navtest.json")] if len(PKGROOT.parents) > 3 else [])
    real = next((c for c in cands if c and Path(c).is_file()), None)
    if real is None:
        skip("the real navtest list is not on this machine (AUX_NAVTEST unset, no repo copy)")
        return
    raw = Path(real).read_bytes()
    ok(hashlib.sha256(raw.replace(chr(13).encode(), b"")).hexdigest() == pin, "the real list is not the pinned one")
    rc, out, _ = TC.bash_run(f'bash "{TC.posix(sc)}" "{TC.posix(real)}"')
    ok(out == pin, out)
    ok(TA.navtest_identity(raw, crlf_rec)[0])
    lf_real, crlf_real = raw.replace(chr(13).encode(), b""), raw.replace(chr(13).encode(), b"").replace(b"\n", chr(13).encode() + b"\n")
    ok(TA.navtest_identity(lf_real, crlf_rec)[0] and TA.navtest_identity(crlf_real, crlf_rec)[0] and hashlib.sha256(crlf_real).hexdigest() == crlf_rec)       # BOTH forms of the real list pass
    ok(not TA.navtest_identity(lf_real.replace(b'"', b"'", 1), crlf_rec)[0])                                # one byte changed: refused


def chain_static():
    # the S1c properties of the chain text (S1's own are check_chain's)
    ok("ARM_ORDER_S1C=\"H1u10k-s0 H1u2k-s0\"" in CHAIN_TEXT and 'ARMS="${S1AUX_ARMS:-$ARM_ORDER_S1}"' in CHAIN_TEXT)                    # the DEFAULT list is S1's nine, never an S1c arm
    ok('--profile "$PROFILE"' in CHAIN_TEXT and CHAIN_TEXT.count('--profile "$PROFILE"') == 2 and "--profile refe_aux_s1 " not in CHAIN_TEXT)
    ok('arm_profile() { case "$1" in H1u*) echo refe_aux_s1c ;; *) echo refe_aux_s1 ;; esac; }' in CHAIN_TEXT and 'PROFILE="$(arm_profile "$ARM")"' in CHAIN_TEXT)
    ok(CHAIN_TEXT.index('PROFILE="$(arm_profile "$ARM")"') < CHAIN_TEXT.index("scorer_finetune.py $ARGS --preflight"))                      # the profile is known before the first use of the gate
    ok("refe/test_s1c.py" in CHAIN_TEXT.split("# ---- 1. env")[0])                                            # the precheck demands the S1c suite script
    ok(CHAIN_TEXT.count("DD_DEFAULT_MB=24576") == 1 and "[ \"$S1C_ONLY\" = 1 ] && DD_DEFAULT_MB=5120" in CHAIN_TEXT)
    ok(CHAIN_TEXT.index("S1C_ONLY=1; for _a in $ARMS") < CHAIN_TEXT.index("upstream() {") < CHAIN_TEXT.index("DD_DEFAULT_MB=24576"))      # S1C_ONLY is defined before its two readers
    # arms_ok over the union: S1c arms are registered, in order, alone or after S1's; an S1 arm after an S1c arm is out of order
    tmp = Path(tmpdir())

    def arms(a):
        sc = tmp / "ao.sh"
        TC.write_sh(sc, TC.header_text(CHAIN_TEXT) + f'\nARMS="{a}"\n' + TC.fn_body(CHAIN_TEXT, "arms_ok") + "\narms_ok\n")
        return TC.bash_run(f'bash "{TC.posix(sc)}"')
    ok(arms("H1u10k-s0")[0] == 0 and arms("H1u2k-s0")[0] == 0 and arms("H1u10k-s0 H1u2k-s0")[0] == 0 and arms("N-s0 H1u10k-s0")[0] == 0)
    ok(arms("H1u2k-s0 H1u10k-s0")[0] != 0 and arms("H1u10k-s0 N-s0")[0] != 0 and arms("H1u10k-s1")[0] != 0 and arms("H1u10k-s0 H1u10k-s0")[0] != 0)
    # the dd size: the chain's own lines under bash (S1c-only -> 5120, any S1 arm -> 24576, the env wins)
    for arms_, env, want in (("H1u10k-s0", {}, 5120), ("H1u10k-s0 H1u2k-s0", {}, 5120), ("N-s0 H1u10k-s0", {}, 24576), ("N-s0 H2-s0", {}, 24576), ("H1u10k-s0", {"S1AUX_DD_MB": "777"}, 777)):
        line = next(ln for ln in CHAIN_TEXT.split("\n") if ln.startswith("DD_DEFAULT_MB=24576"))
        s1c_line = next(ln for ln in CHAIN_TEXT.split("\n") if ln.startswith("S1C_ONLY=1; for _a in $ARMS"))
        sc = tmp / "dd.sh"
        TC.write_sh(sc, f'ARMS="{arms_}"\n{s1c_line}\n{line}\nDDMB="${{S1AUX_DD_MB:-$DD_DEFAULT_MB}}"\necho "$DDMB"\n')
        rc, out, err = TC.bash_run(f'bash "{TC.posix(sc)}"', env=env)
        ok(rc == 0 and out == str(want), (arms_, env, out, err))


def upstream_cases(text, tmp, s1c_only):
    """The chain's gpu_busy / upstream / wait_clear, extracted VERBATIM from `text` and run under bash (flock shimmed, nvidia-smi faked by a call counter, exactly as test_s1aux_chain does).
    -> {case: (rc, stdout, chain.log)}.  The RL-S1 states under test: its chain.lock HELD (R0c training), FREE with no ZZRLS1_DONE (a K3-killed R0c), FREE with ZZRLS1_DONE, FREE with no chain.log at all."""
    tmp = Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    p2, rl = tmp / "p2", tmp / "rls1"
    p2.mkdir()
    rl.mkdir()
    nv = tmp / "nv.sh"
    TC.write_sh(nv, '#!/bin/bash\nn=$(cat "$NVCOUNT" 2>/dev/null || echo 0)\necho $((n + 1)) > "$NVCOUNT"\nif [ "$n" -lt "${NV_BUSY_CALLS:-0}" ]; then echo 4242; fi\nexit "${NV_RC:-0}"\n')
    harness = tmp / "wc.sh"
    TC.write_sh(harness, f'P2="{TC.posix(p2)}"; RLS1="{TC.posix(rl)}"; LOGF="{TC.posix(tmp / "wc.log")}"\n' + ("S1C_ONLY=1\n" if s1c_only else "") +
                'log() { echo "$*" >> "$LOGF"; }\nSTAGE=precheck\nrefuse() { echo "REFUSED $*" >> "$LOGF"; exit 1; }\nflock() { return "${FAKE_FLOCK_RC:-0}"; }\n'
                + TC.fn_body(text, "gpu_busy") + TC.fn_body(text, "upstream") + TC.fn_body(text, "wait_clear") + 'wait_clear\necho CLEAR\n')

    def wc(**env):
        (tmp / "wc.log").write_text("")
        (tmp / "nvc").write_text("0")
        e = {"S1AUX_NVSMI": TC.posix(nv), "NVCOUNT": TC.posix(tmp / "nvc"), "S1AUX_POLL_S": "0", "S1AUX_IDLE_POLLS": "3", **env}
        rc, out, err = TC.bash_run(f'bash "{TC.posix(harness)}"', e, timeout=60)
        return rc, out, (tmp / "wc.log").read_text()
    res = {}
    bounded = {"FAKE_FLOCK_RC": "1", "S1AUX_POLL_S": "1", "S1AUX_WAIT_MAX_S": "1"}              # a held lock is probed as held (flock -n fails); the wait is bounded so the test ends
    (rl / "chain.log").write_text("ZZRLS1_ARM_DONE R0\n")                                          # an R0c killed by K3: arms logged, NO ZZRLS1_DONE
    (rl / "chain.lock").write_text("")
    res["held_no_done"] = wc(**bounded)                                                            # R0c TRAINING: WAIT (and never the 'neither running nor finished' refusal)
    (rl / "chain.lock").unlink()
    res["free_no_done"] = wc()                                                                     # R0c over, killed: PROCEED with the WARN
    res["free_no_done_abandoned"] = wc(S1AUX_RLS1_ABANDONED="1")
    (rl / "chain.log").write_text("ZZRLS1_ARM_DONE R0\nZZRLS1_DONE arms: R0c\n")
    res["free_done"] = wc()                                                                        # R0c over, finished: PROCEED, no WARN
    (rl / "chain.log").unlink()
    res["free_no_log"] = wc()                                                                      # no RL-S1 log at all (it never ran on this pod): PROCEED with the WARN
    (rl / "chain.log").write_text("ZZRLS1_ARM_DONE R0\nZZRLS1_DONE arms: R0c\n")            # from here the RL-S1 part is satisfied in BOTH modes: these cases isolate P2's lock and the GPU polls
    (p2 / "chain.lock").write_text("")
    res["p2_held"] = wc(**bounded)                                                                 # P2's lock is still waited on
    (p2 / "chain.lock").unlink()
    res["gpu_busy_twice"] = wc(NV_BUSY_CALLS="2")                                                  # the idle-GPU polls are as before: busy twice, then 3 consecutive idle polls
    res["gpu_busy_twice_polls"] = int((tmp / "nvc").read_text())
    res["gpu_never"] = wc(NV_BUSY_CALLS="1000", S1AUX_POLL_S="1", S1AUX_WAIT_MAX_S="2")
    return res


def upstream_s1c():
    base = Path(tmpdir()) / "up"
    WARN = "ZZS1AUX_UPSTREAM_NO_DONE rls1"

    def judged(text, s1c_only):
        r = upstream_cases(text, base, s1c_only)
        v = []
        rc, out, lg = r["held_no_done"]
        v.append(("held waits", rc == 1 and "RL-S1 chain still holds" in lg and "ZZS1AUX_WAITING" in lg and "neither running nor finished" not in lg and WARN not in lg))
        rc, out, lg = r["free_no_done"]
        v.append(("free no DONE proceeds + WARN", rc == 0 and out == "CLEAR" and WARN in lg and "REFUSED" not in lg and "neither running nor finished" not in lg))
        v.append(("free no DONE, abandoned, proceeds", r["free_no_done_abandoned"][0] == 0 and WARN not in r["free_no_done_abandoned"][2]))
        rc, out, lg = r["free_done"]
        v.append(("free DONE proceeds, no WARN", rc == 0 and out == "CLEAR" and WARN not in lg and "REFUSED" not in lg))
        rc, out, lg = r["free_no_log"]
        v.append(("free no log proceeds + WARN", rc == 0 and out == "CLEAR" and WARN in lg and "REFUSED" not in lg))
        rc, out, lg = r["p2_held"]
        v.append(("P2 lock waits", rc == 1 and "P2 chain still holds" in lg and "ZZS1AUX_WAITING" in lg))
        rc, out, lg = r["gpu_busy_twice"]
        v.append(("idle polls", rc == 0 and out == "CLEAR" and lg.count("ZZS1AUX_WAITING") == 2 and r["gpu_busy_twice_polls"] == 5))      # 2 busy polls + 3 consecutive idle polls
        rc, out, lg = r["gpu_never"]
        v.append(("a GPU that never frees: bounded", rc == 1 and "REFUSED waited" in lg))
        return v
    # an S1c-only launch: every case as the PI asked (R0c lock held => waits; lock free + no DONE => proceeds with the WARN; lock free + DONE => proceeds)
    for name, good in judged(CHAIN_TEXT, True):
        ok(good, name)
    # S1's own nine-arm chain keeps its rule (S1-D4): lock free + no DONE is a REFUSAL there, with the lock held still a wait
    s1 = dict(judged(CHAIN_TEXT, False))
    ok(s1["held waits"] and s1["free DONE proceeds, no WARN"] and s1["free no DONE, abandoned, proceeds"] and s1["P2 lock waits"] and s1["idle polls"], s1)
    ok(not s1["free no DONE proceeds + WARN"] and not s1["free no log proceeds + WARN"], s1)
    rc, out, lg = upstream_cases(CHAIN_TEXT, base, False)["free_no_done"]
    ok(rc == 1 and "neither running nor finished" in lg, lg)
    # DELIBERATE REGRESSION: restoring the old refusal in S1c's chain must turn the S1c cases RED
    old = mutate(CHAIN_TEXT, 'if [ "${S1C_ONLY:-0}" = 1 ]; then\n      log "ZZS1AUX_UPSTREAM_NO_DONE', 'if false; then\n      log "ZZS1AUX_UPSTREAM_NO_DONE')
    red = [name for name, good in judged(old, True) if not good]
    ok("free no DONE proceeds + WARN" in red and "free no log proceeds + WARN" in red and len(red) == 2, red)
    # and dropping the WARN (silent proceed) is RED too; dropping the lock wait (RL-S1 lock ignored) is RED
    quiet = mutate(CHAIN_TEXT, 'log "ZZS1AUX_UPSTREAM_NO_DONE rls1', 'true "ZZS1AUX_UPSTREAM_NO_DONE_QUIET rls1')
    ok({n for n, g in judged(quiet, True) if not g} == {"free no DONE proceeds + WARN", "free no log proceeds + WARN"})
    nolock = mutate(CHAIN_TEXT, 'if [ -e "$RLS1/chain.lock" ] && ! ( flock -n 9 ) 9<> "$RLS1/chain.lock"; then echo "the RL-S1 chain still holds $RLS1/chain.lock"; return 2; fi', ":")
    ok("held waits" in {n for n, g in judged(nolock, True) if not g})


# ============================================================================================ 8. the REAL chain, end to end, S1c arm only
S1C_STUB_TAIL = r'''
ups = int(val("--max-updates") or 400); every = int(val("--eval-every") or 400); bs = int(val("--batch") or 8); acc = int(val("--accum") or 8)
run_n = ups // 2 if os.environ.get("FAKE_S1C_EARLY_END") == arm else ups              # the epoch cap (TL-1): the run ends silently at half the registered updates
tags = ["step0"] + ["update%d" % u for u in range(every, run_n + 1, every)] + (["update%d" % run_n] if run_n % every else [])
L = [data, {"event": "start", "updates": run_n, "micro_per_update": acc, "batch": bs}]
for u in sorted({25, run_n // 2, run_n}):
    r = {"event": "train", "update": u, "of": run_n, "bceA": 0.5, "lossB": 0.4, "lr": 0.0 if u == run_n else 1e-4}
    if head: r["aux_bce"] = 0.69
    L.append(r)
for tag in tags:
    L.append({"event": "eval", "tag": tag, "n": 3137})
    if head: L.append({"event": "aux_eval", "tag": tag, "iou": 0.5})
L.append({"event": "done", "updates": run_n})
with open(os.path.join(out, "sft.jsonl"), "a") as f:
    for r in L: f.write(json.dumps(r) + "\n")
open(os.path.join(out, "model_sft_B.pt"), "wb").write(b"B" * 4096)
os.makedirs(os.path.join(out, "dump"), exist_ok=True)
for tag in tags:
    for k in ("base", "A", "B"): open(os.path.join(out, "dump", "scores_%s_%s.npz" % (tag, k)), "wb").write(b"s")
    if head: open(os.path.join(out, "dump", "head_pred_%s.npz" % tag), "wb").write(b"h")
if head: open(os.path.join(out, "aux_drv.pt"), "wb").write(b"aux")
open(os.path.join(out, "scorer_states_last.pt"), "wb").write(b"states")
print("ZZSFT_DONE")
'''


def end_to_end(tmp):
    import shlex
    marker = 'ups = int(os.environ.get("FAKE_UPDATES", "400"))'
    saved = (TC.STUB_SCORER, TC.STUB_GATE)
    assert saved[0].count(marker) == 1 and saved[1].count('val("--profile") == "refe_aux_s1"') == 1
    TC.STUB_SCORER = saved[0].split(marker)[0] + S1C_STUB_TAIL
    TC.STUB_GATE = saved[1].replace('val("--profile") == "refe_aux_s1"', 'val("--profile") in ("refe_aux_s1", "refe_aux_s1c")')
    keep = SC.MODEL_B_MIN_BYTES
    try:
        if os.name == "nt" and subprocess.run([shutil.which("bash"), "-c", "exec 200>/dev/null"], capture_output=True).returncode != 0:
            return skip("end_to_end: this bash cannot open fd 200")
        if not TC.NAVTEST.is_file() or not (TC.REAL_STORE / "manifest.json").is_file():          # neither is in the shipped package: on the pod this test is SKIPPED (not counted), exactly as test_s1aux_chain's sandbox is
            return skip("end_to_end: the repo's navtest list or the real aux label store is absent on this machine (the sandbox copies both)")
        sb = TC.Sandbox(tmp, CHAIN_TEXT)
        if sb.store != TC.REAL_STORE:
            return skip("end_to_end: the real aux label store is absent on this machine (the chain stages past `data` need it)")
        rl = sb.W / "data" / "rls1"
        (rl / "chain.log").write_text("ZZRLS1_ARM_DONE R0\n")                                  # R0c ended WITHOUT ZZRLS1_DONE (a K3 kill), and holds no lock: the 2026-10-14 case
        E = {"S1AUX_ARMS": "H1u10k-s0"}
        rc, out, lg = sb.run(fresh=True, env_extra=E)
        ok(rc == 0 and "ZZS1AUX_DONE arms: H1u10k-s0" in lg, (rc, lg[-900:], out[-300:]))
        seg = lg[lg.rfind("ZZS1AUX_START"):]
        ok("ZZS1AUX_UPSTREAM_NO_DONE rls1" in seg and "WARN" in seg, seg[:600])                  # proceeded, and SAID so
        starts = re.findall(r"ZZS1AUX_ARM_TRAIN_START (\S+)", seg)
        ok(starts == ["H1u10k-s0"], starts)                                                        # the chain ran ONLY the S1c arm
        marks = sb.marks()
        ok("arm_H1u10k-s0" in marks and not [m for m in marks if m.startswith("arm_") and m != "arm_H1u10k-s0"], marks)
        sc = sb.calls("scorer_finetune")
        want = shlex.split(TC.arm_args_of(sb.text, "H1u10k-s0", sb.tmp)[1])
        ok([c["arm"] for c in sc] == ["H1u10k-s0", "H1u10k-s0"] and [c["preflight"] for c in sc] == [True, False], [(c["arm"], c["preflight"]) for c in sc])
        ok(sc[0]["argv"] == want + ["--preflight"] and sc[1]["argv"] == want, sc[1]["argv"])
        ok(all(tok in want for tok in ("--max-updates", "10000", "--batch", "8", "--accum", "1", "--eval-every", "5000")))
        ok(not any(c["hf_token"] for c in sc))
        gl = sb.calls("launch_gate")
        ok(gl and all(c["profile"] == "refe_aux_s1c" for c in gl) and [c["cmd"] for c in gl] == ["run", "verify", "verify"], [(c["cmd"], c["profile"]) for c in gl])
        run_c = [c for c in gl if c["cmd"] == "run"][0]
        ok(shlex.split(run_c["args"]) == want and run_c["arm"] == "H1u10k-s0")
        arm_files = sb.hub / "Sayood_tanitad-refe-s1aux" / "arms" / "H1u10k-s0"
        names = {p_.relative_to(arm_files).as_posix() for p_ in arm_files.rglob("*") if p_.is_file()}
        need = {"model_sft_B.pt", "aux_drv.pt", "sft.jsonl", "run_check.json", "dump/head_pred_step0.npz", "dump/head_pred_update5000.npz", "dump/head_pred_update10000.npz",
                "dump/scores_update5000_B.npz", "dump/scores_update10000_B.npz"}
        ok(need <= names and "model_sft_A.pt" not in names, sorted(need - names))
        ok(not (sb.hub / "Sayood_tanitad-refe-s1aux" / "arms" / "N-s0").exists())
        # the second budget: S1AUX_ARMS names it, nothing else changes (accum 8, update 2,000, evals every 1,000)
        rc, out, lg = sb.run(env_extra={"S1AUX_ARMS": "H1u2k-s0"})
        ok(rc == 0 and "ZZS1AUX_DONE arms: H1u2k-s0" in lg, (rc, lg[-600:]))
        w2 = shlex.split(TC.arm_args_of(sb.text, "H1u2k-s0", sb.tmp)[1])
        t2 = [c for c in sb.calls("scorer_finetune") if c["arm"] == "H1u2k-s0" and not c["preflight"]]
        ok(len(t2) == 1 and t2[0]["argv"] == w2 and w2[w2.index("--max-updates") + 1] == "2000" and w2[w2.index("--accum") + 1] == "8")
        ok((sb.hub / "Sayood_tanitad-refe-s1aux" / "arms" / "H1u2k-s0" / "dump" / "head_pred_update2000.npz").is_file())
        # DELIBERATE REGRESSION: the trainer ends early (the epoch cap, TL-1) with rc 0 and ZZSFT_DONE => the chain's run-check REFUSES the arm and banks nothing for it
        for d in (sb.R / "runs" / "H1u10k-s0", sb.R / "lg" / "H1u10k-s0"):
            shutil.rmtree(d, ignore_errors=True)
        for m_ in (sb.R / "stage").glob("*_H1u10k-s0.ok"):
            m_.unlink()
        shutil.rmtree(sb.hub / "Sayood_tanitad-refe-s1aux" / "arms" / "H1u10k-s0", ignore_errors=True)
        rc, out, lg = sb.run(env_extra={**E, "FAKE_S1C_EARLY_END": "H1u10k-s0"})
        ok(rc == 1 and "ZZS1AUX_REFUSED stage=arm_H1u10k-s0:" in lg and "is not a valid S1 arm" in lg and "want exactly one reading updates == 10000" in lg, (rc, lg[-700:]))
        ok(not (sb.hub / "Sayood_tanitad-refe-s1aux" / "arms" / "H1u10k-s0").exists() and "arm_H1u10k-s0" not in sb.marks())
    finally:
        TC.STUB_SCORER, TC.STUB_GATE = saved
        SC.MODEL_B_MIN_BYTES = keep


def main() -> int:
    tmp = tmpdir()
    tables()
    P, argv_of = gate(tmp)
    dvb(argv_of)
    run_check_budgets()
    chain_static()
    upstream_s1c()
    navtest_pin(tmp)
    if "--no-e2e" in sys.argv[1:]:                                                              # the gate's POD run (launch_gate.AUXS1C_SUITE): a DECLARED scope, not a skip
        print("ZZTEST_SCOPE end_to_end NOT run (--no-e2e): the chain sandbox needs the repo checkout -- dev-box-only, like test_s1aux_chain's sandbox")
    else:
        end_to_end(tmp)
    reachability()
    if os.environ.get("S1C_EVIDENCE_OUT"):
        Path(os.environ["S1C_EVIDENCE_OUT"]).write_text(json.dumps({"tool": "refe/test_s1c.py reachability", "checks_passed": N_OK, "evidence": EVID}, indent=1, default=str), encoding="utf-8")
    print(f"ZZTEST_PASS {N_OK}" + (f"  ({len(SKIPS)} skipped, not counted)" if SKIPS else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
