#!/usr/bin/env python3
"""lit1_selftest.py -- the self-test and MUTATION pass of the LAUNCH-INTENT T1-kin chain (lit1_chain.sh + lit1_lib.py + lit1_tree.py + wake_lit1.sh).  CPU only, no GPU, no rollout, no network, no ssh, no git write.

    python lit1_selftest.py [--only a,b] [--no-mutation] [--workers N] [--out FILE]

Three layers, all inside F:/TanitAD-scratch/launch_intent_t1 (the only place this file writes):
  A  PYTHON LAYER (this file): the box-2 log parser and run states, pullcheck COMPLETE-BY-CONTENT, the ~45 run-record clauses, guards, world_ok, templates and the banked-argv proof, commits, the tool-tree and AST proofs,
     rolloutcode and K-BUILD (on a fake object store as well as the real mirror), static scans of the shipped files.
  B  SHELL LAYER (lit1_selftest_sh.py): the real lit1_chain.sh run to its end markers in a throw-away WORLD (synthetic pulls + box-2 log, a STUB tool that writes the files the real tool writes, a STUB reader).  EVERY shell step goes through
     lit1_sealed.run: a guard inside the same bash makes the world's ssh / scp stubs the only ones, HOME the empty scratch HOME and the box-2 alias unresolved BEFORE the step is parsed (the 2026-10-28 05:56 incident).
  C  MUTATION PASS: every `on(clause)` check is switched off one at a time, and ~100 textual mutants of the shipped code are applied one at a time; the scenario that owns each one MUST then fail.  A mutant that survives is a finding.
Evidence class of every line this prints: MEASURED (this run).  Not a rollout, not a science result.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import lit1_fix as X  # noqa: E402
import lit1_lib as L  # noqa: E402
import lit1_sealed as S  # noqa: E402
import lit1_tree as T  # noqa: E402

PYW = f"{L.SCRATCH}/w_py"
MIRROR = "C:/Users/Admin/tanitad-push"
SHIPPED = ["lit1_chain.sh", "lit1_lib.py", "lit1_tree.py", "wake_lit1.sh", "lit1_chain.cmd", "argv/rollout_causal.tmpl", "argv/rollout_logged.tmpl", "argv/t0fresh.tmpl", "argv/reader.tmpl"]


Chk = X.Chk


@contextlib.contextmanager
def dis(Lm, *ids):
    old = set(Lm.DISABLED)
    Lm.DISABLED.update(ids)
    try:
        yield
    finally:
        Lm.DISABLED.clear()
        Lm.DISABLED.update(old)


@contextlib.contextmanager
def envset(**kw):
    old = {k: os.environ.get(k) for k in kw}
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def load_pair(d: str):
    """Import lit1_lib / lit1_tree FROM directory d as a fresh pair (a mutated copy), leaving sys.modules as it was."""
    saved = {k: sys.modules.get(k) for k in ("lit1_lib", "lit1_tree")}
    try:
        for k in saved:
            sys.modules.pop(k, None)
        sys.path.insert(0, d)
        Lm = importlib.import_module("lit1_lib")
        Tm = importlib.import_module("lit1_tree")
        return Lm, Tm
    finally:
        sys.path.remove(d)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# ================================================================================================================================ A. the run-record clauses (one violation per clause id)
def refix(P):
    sha = hashlib.sha256(" ".join(P["argv"]["argv"]).encode()).hexdigest()[:16]
    P["argv"]["sha"] = P["summary"]["argv_sha"] = P["selfcheck"]["argv_sha"] = sha


def a_set(name, val):
    def f(P):
        a = P["argv"]["argv"]
        a[a.index(name) + 1] = str(val)
        refix(P)
    return f


def a_add(name, val):
    def f(P):
        P["argv"]["argv"].extend([name, str(val)])
        refix(P)
    return f


def s_set(path, val):
    def f(P):
        d = P
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = val
    return f


def counts(n, e):
    def f(P):
        sm = P["summary"]
        sm["launch_n"], sm["launch_eligible"], sm["launch_rows"] = n, e, e
        sm["launch_weight_share"] = L.share_of(n, e, L.LI_W)
    return f


def _del_share(P):
    del P["summary"]["launch_weight_share"]


H64 = "ab" * 32
CASES = {      # clause id -> (arm, seed, violation, clause ids switched off ALONGSIDE it in the mutant run's control)
    "argv-steps": ("launchintent", 0, a_set("--steps", 2999)), "argv-dagger-p": ("launchintent", 0, a_set("--dagger-p", 0.2)), "argv-w-acc": ("launchintent", 0, a_set("--w-acc", 40)),
    "argv-seed": ("launchintent", 0, a_set("--seed", 5)), "argv-dagger-seed": ("launchintent", 0, a_set("--dagger-seed", 1)), "argv-bank": ("launchintent", 0, a_set("--dagger-bank-sha256", H64)),
    "argv-init": ("launchintent", 0, a_set("--init-ckpt", "D:\\x\\M2b_s1\\ckpt_last.pt")), "argv-out": ("launchintent", 0, a_set("--out", "D:\\x\\M2b_s0_dagoacc")),
    "argv-lever": ("launchintent", 0, a_set("--launch-intent-w", 2.0)), "argv-lever-once": ("launchintent", 0, a_add("--launch-intent-w", "1.58926")),
    "argv-no-ego": ("launchintent", 0, a_set("--ego-perturb-p", 0.5)), "argv-sha": ("launchintent", 0, s_set(["argv", "sha"], "0" * 16)),
    "sum-done": ("launchintent", 0, s_set(["summary", "done"], False)), "sum-steps": ("launchintent", 0, s_set(["summary", "steps"], 2999)), "sum-seed": ("launchintent", 0, s_set(["summary", "seed"], 9)),
    "sc-pass": ("launchintent", 0, s_set(["selfcheck", "declared_vs_built"], "FAIL")), "sum-lever": ("launchintent", 0, s_set(["summary", "levers", "launch_intent_w"], 0.0)),
    "sum-lever-mask": ("launchintent", 0, s_set(["summary", "levers", "launch_intent", "v0_max"], 5.0)), "sum-keys": ("launchintent", 0, _del_share),
    "sum-top-w": ("launchintent", 0, s_set(["summary", "launch_intent_w"], 1.0)), "sum-share": ("launchintent", 0, counts(57000, 191311)),
    "sum-share-rec": ("launchintent", 0, s_set(["summary", "launch_weight_share"], 0.5)), "sum-rows": ("launchintent", 0, s_set(["summary", "launch_rows"], 191310)),
    "sum-mask": ("launchintent", 0, counts(38262, 191311)),
    "sum-window": ("launchintent", 0, s_set(["summary", "train_window_list_sha256"], H64)), "sum-init": ("launchintent", 0, s_set(["summary", "levers", "init_ckpt", "sha256"], H64)),
    "sum-init-applied": ("launchintent", 0, s_set(["summary", "levers", "init_ckpt", "applied"], False)), "sum-bank": ("launchintent", 0, s_set(["summary", "levers", "dagger_bank_sha256"], H64)),
    "sum-dagger-p": ("launchintent", 0, s_set(["summary", "levers", "dagger_p"], 0.2)), "sum-w-acc": ("launchintent", 0, s_set(["summary", "levers", "w_acc"], 1.0)),
    "sum-ego-zero": ("launchintent", 0, s_set(["summary", "levers", "ego_perturb", "ego_perturb_p"], 0.5)),
    "tok-kind": ("launchintent", 0, s_set(["token", "verdict"], "FAIL")), "tok-run": ("launchintent", 0, s_set(["token", "run"], "M2b_s0_dagoacc")),
    "tok-commit": ("launchintent", 0, s_set(["token", "binding", "commit"], "abc")), "tok-shas": ("launchintent", 0, s_set(["token", "reader_sha256"], "zz")),
    "tok-trainer": ("launchintent", 0, s_set(["token", "key_files_sha256", L.TRAINER_PATH], H64)), "tok-model": ("launchintent", 0, s_set(["token", "key_files_sha256", L.MODEL_PATH], H64)),
    "tok-init": ("launchintent", 0, s_set(["token", "init_ckpt_sha256"], H64)), "tok-bank": ("launchintent", 0, s_set(["token", "bank_sha256"], H64)), "tok-wacc": ("launchintent", 0, s_set(["token", "w_acc"], 1.0)),
    "tok-excluded": ("launchintent", 0, s_set(["token", "excluded_sha256"], H64)), "tok-arm": ("launchintent", 0, s_set(["token", "arm"], "launchintentrep")),
    "ck-not-ctrl": ("launchintent", 0, None),
}
ALONE = {"sum-mask": ("sum-share",)}      # sum-mask is SUBSUMED by sum-share (every n/e that passes the 1/3 +- 0.02 share also passes f +- 0.02): it is tested with sum-share switched off in BOTH runs (NOTES: equivalent as a gate, kept as the registered K-MASK)
# a violation that is ALSO caught by another clause is reported as COVERED (still killed: its own clause id disappears from the message); SOLE means the clause alone refuses it
PULL_IDS = ["pull-age", "pull-names", "pull-rehash", "pull-done", "pull-3way"]
LOG_IDS = ["log-cfail", "log-path", "log-label", "log-stop-after"]
GUARD_IDS = ["guard-ram", "guard-disk"]
TREE_IDS = ["tree-set", "tree-onefile", "tree-extras", "tree-overlay", "tree-base-digest", "proof-removed", "proof-added", "proof-lines"]
KB_IDS = ["kb-compose-bytes", "kb-launch-bytes", "kb-diffpatch", "kb-green", "kb-mutation", "kb-flagoff-test", "kb-golden", "kb-default-off"]


def t_clause(Lm, Tm, cid):
    c = Chk()
    arm, seed, mut = CASES[cid]
    art = f"{PYW}/art_{cid}"
    good = X.mk_run(f"{art}_ok", arm, seed)
    off = ALONE.get(cid, ())
    with dis(Lm, *off):
        c.eq(Lm.runrecord_problems(good, arm, seed, Lm.sha256_file(good + "/ckpt_last.pt")), [], f"{cid}: the clean fixture must have no problem")
        bad = X.mk_run(art, arm, seed, mut=mut)
        ck = Lm.OACC_CKPT_SHA[0] if cid == "ck-not-ctrl" else Lm.sha256_file(bad + "/ckpt_last.pt")
        probs = Lm.runrecord_problems(bad, arm, seed, ck)
        c.ok(any(f"[{cid}]" in p for p in probs), f"{cid}: the violating fixture was NOT refused by [{cid}] (problems: {probs[:3]})")
    return c.fails


def t_clause_positive(Lm, Tm):
    c = Chk()
    for (arm, seed) in L.RUNS:
        d = X.mk_run(f"{PYW}/art_pos", arm, seed)
        rc, kind, msg = Lm.pullcheck(d, arm, seed, expect_sha=Lm.sha256_file(d + "/ckpt_last.pt"))
        c.eq((rc, kind), (0, "COMPLETE"), f"{L.LABEL[(arm, seed)]} clean pull")
        c.has(msg, L.LABEL[(arm, seed)], "label in message")
    return c.fails


def t_pull(Lm, Tm, cid):
    c = Chk()
    art = f"{PYW}/art_{cid}"
    arm, seed = "launchintent", 0
    run = L.run_name(arm, seed)
    if cid == "pull-age":
        d = X.mk_run(art, arm, seed, age=0.0)
        rc, kind, msg = Lm.pullcheck(d, arm, seed, min_age=3.0)
        c.eq((rc, kind), (1, "INCOMPLETE"), "a SHA256SUMS.txt younger than min-age")
        c.has(msg, "younger", "the reason")
    elif cid == "pull-names":
        d = X.mk_run(art, arm, seed, sums=False)
        X.write_sums(d, L.eval_tag(arm, seed), names=L.sums_names(L.eval_tag(arm, seed))[:-1])
        rc, kind, msg = Lm.pullcheck(d, arm, seed)
        c.eq((rc, kind), (1, "INCOMPLETE"), "SHA256SUMS.txt lists 7 of the 8 names")
        c.has(msg, "differently", "the reason")
    elif cid == "pull-rehash":
        d = X.mk_run(art, arm, seed)
        open(d + "/train_log.jsonl", "a").write("{}\n")
        rc, kind, msg = Lm.pullcheck(d, arm, seed)
        c.eq((rc, kind), (1, "INCOMPLETE"), "a file that no longer hashes to its listed sha256")
        c.has(msg, "does not hash", "the reason")
    elif cid == "pull-done":
        d = X.mk_run(art, arm, seed, mut=s_set(["summary", "done"], False))
        rc, kind, msg = Lm.pullcheck(d, arm, seed)
        c.eq((rc, kind), (1, "INCOMPLETE"), "summary.json done is false: INCOMPLETE (a pull still filling), not WRONG")
        c.has(msg, "done", "the reason")
    elif cid == "pull-3way":
        d = X.mk_run(art, arm, seed)
        rc, kind, msg = Lm.pullcheck(d, arm, seed, expect_sha="0" * 64)
        c.eq((rc, kind), (3, "WRONG"), "the box-2 log's checkpoint sha256 is not the pulled file's")
        c.has(msg, "box-2 log printed", "the reason")
    return c.fails


def t_pull_misc(Lm, Tm):
    c = Chk()
    arm, seed = "launchintent", 1
    tag = L.eval_tag(arm, seed)
    art = f"{PYW}/art_pmisc"
    d = f"{art}/{L.run_name(arm, seed)}"
    c.eq(Lm.pullcheck(f"{art}/absent", arm, seed)[:2], (1, "INCOMPLETE"), "absent directory")
    d = X.mk_run(art, arm, seed, sums=False)
    c.eq(Lm.pullcheck(d, arm, seed)[:2], (1, "INCOMPLETE"), "no SHA256SUMS.txt yet")
    X.write_sums(d, tag)
    c.eq(Lm.pullcheck(d, arm, seed)[:2], (0, "COMPLETE"), "the clean pull")
    raw = open(d + "/SHA256SUMS.txt", "rb").read()
    open(d + "/SHA256SUMS.txt", "wb").write(raw.rstrip(b"\n"))      # not newline-terminated: a write still in progress
    c.eq(Lm.pullcheck(d, arm, seed, min_age=0)[:2], (1, "INCOMPLETE"), "SHA256SUMS.txt without a final newline")
    open(d + "/SHA256SUMS.txt", "wb").write(raw.replace(b"\n", b"\r\n"))
    c.eq(Lm.pullcheck(d, arm, seed, min_age=0)[:2], (0, "COMPLETE"), "CRLF lines are accepted")
    open(d + "/SHA256SUMS.txt", "wb").write(raw + raw.split(b"\n")[0] + b"\n")
    c.eq(Lm.pullcheck(d, arm, seed, min_age=0)[:2], (3, "WRONG"), "a name listed twice")
    open(d + "/SHA256SUMS.txt", "wb").write(raw + b"not a sums line\n")
    c.eq(Lm.pullcheck(d, arm, seed, min_age=0)[:2], (3, "WRONG"), "a malformed line")
    open(d + "/SHA256SUMS.txt", "wb").write(raw)
    open(d + "/train_log.jsonl", "wb").write(b"")      # empty file, listed with the sha256 of nothing
    X.write_sums(d, tag)
    c.eq(Lm.pullcheck(d, arm, seed)[:2], (1, "INCOMPLETE"), "an empty file is not a complete pull")
    open(d + "/train_log.jsonl", "w").write('{"step": 1}\n')
    X.write_sums(d, tag)
    os.remove(d + "/eval_" + tag + ".npz")
    c.eq(Lm.pullcheck(d, arm, seed)[:2], (1, "INCOMPLETE"), "a listed file that is missing")
    return c.fails


# ================================================================================================================================ A. the box-2 log
def _sha(i):
    return ("%02x" % (i + 1)) * 32


def LOGART():
    return f"{PYW}/art_log"


def lg(lines):
    return "\n".join(lines) + "\n"


def runs_lines(order=None, over=None):
    out = []
    for (a, s) in (order or L.RUNS):
        out += X.log_run(LOGART(), a, s, _sha(L.RUNS.index((a, s))), **(over or {}).get((a, s), {}))
    return out


def t_log_misc(Lm, Tm):
    c = Chk()
    art = LOGART()
    P = Lm.parse_boxlog(lg(runs_lines() + [X.done_line()]), art)
    for i, (a, s) in enumerate(L.RUNS):
        c.eq(Lm.run_state(P, a, s), ("PASS", _sha(i)), f"{L.LABEL[(a, s)]} state")
    c.eq(Lm.next_ready(P, set()), ("RUN", "launchintent", 0, _sha(0)), "next: the earliest pull")
    c.eq(Lm.next_ready(P, {"M2b_s0_launchintent"}), ("RUN", "launchintent", 1, _sha(1)), "next with A_0 complete is A_1 (a complete run is never returned again)")
    c.eq(Lm.next_ready(P, {L.run_name(a, s) for (a, s) in L.RUNS})[0], "WAIT", "next with everything complete waits")
    m = Lm.log_meta(P)
    c.eq((m["done"], m["pending"], m["pulled"], m["stops"]), (1, 0, 4, 0), "meta of a finished chain")
    # pull order, not the box-2 queue order
    order = [("launchintentrep", 0), ("launchintent", 1), ("launchintent", 0), ("launchintentrep", 1)]
    P2 = Lm.parse_boxlog(lg(runs_lines(order) + [X.done_line()]), art)
    c.eq(Lm.pull_order(P2), order, "pull order follows the log lines")
    c.eq(Lm.next_ready(P2, set())[1:3], ("launchintentrep", 0), "the run whose pull landed first is processed first")
    # a CRLF log is the same log
    P3 = Lm.parse_boxlog(lg(runs_lines() + [X.done_line()]).replace("\n", "\r\n"), art)
    c.eq(Lm.run_state(P3, "launchintent", 0)[0], "PASS", "CRLF log")
    # awaiting controls: a pulled run with only its run record is NOT ready
    aw = X.log_run(art, "launchintent", 0, _sha(0), ksh=None) + X.log_run(art, "launchintent", 1, _sha(1))
    P4 = Lm.parse_boxlog(lg(aw), art)
    c.eq(Lm.run_state(P4, "launchintent", 0)[0], "AWAITING-CONTROLS", "run record without K-SHARE")
    c.eq(Lm.next_ready(P4, set())[:3], ("RUN", "launchintent", 1), "an awaiting run is skipped for a later PASS run")
    P4b = Lm.parse_boxlog(lg(X.log_run(art, "launchintent", 0, _sha(0), rec=None, ksh=None)), art)
    c.eq(Lm.next_ready(P4b, set())[0], "WAIT", "a pull with no controls at all waits")
    # a re-pull restarts the section: the LAST pull line wins
    rp = X.log_run(art, "launchintent", 0, _sha(0)) + X.log_run(art, "launchintent", 0, _sha(7), rec=None, ksh=None)
    P5 = Lm.parse_boxlog(lg(rp), art)
    c.eq(Lm.run_state(P5, "launchintent", 0)[0], "AWAITING-CONTROLS", "a re-pull without controls is not the earlier PASS")
    # control lines of ANOTHER run inside this run's section
    cross = X.log_run(art, "launchintent", 0, _sha(0), ksh=None) + [X.log_run(art, "launchintent", 1, _sha(1))[2]]
    P6 = Lm.parse_boxlog(lg(cross), art)
    c.eq(Lm.run_state(P6, "launchintent", 0)[0], "FAIL", "a K-SHARE line of another run inside A_0's section")
    cross2 = X.log_run(art, "launchintent", 0, _sha(0), rec=None, ksh=None) + [X.log_run(art, "launchintent", 1, _sha(1))[1]] + [X.log_run(art, "launchintent", 0, _sha(0))[2]]
    P7 = Lm.parse_boxlog(lg(cross2), art)
    c.ok(Lm.run_state(P7, "launchintent", 0)[0] != "PASS", "a run record of another run can never make A_0 PASS")
    c.ok(len(P7["problems"]) >= 1, "the parser reports the stray run-record line")
    # a non-PASS record or K-SHARE is a FAIL
    P8 = Lm.parse_boxlog(lg(runs_lines(over={("launchintent", 0): {"rec": "[argv-lever]"}})), art)
    c.eq(Lm.run_state(P8, "launchintent", 0)[0], "FAIL", "run record not PASS")
    P9 = Lm.parse_boxlog(lg(runs_lines(over={("launchintent", 0): {"ksh": "FAIL"}})), art)
    c.eq(Lm.run_state(P9, "launchintent", 0)[0], "FAIL", "K-SHARE FAIL")
    c.eq(Lm.next_ready(P9, set())[0], "FAIL", "next surfaces the FAIL")
    P8b = Lm.parse_boxlog(lg(X.log_run(art, "launchintent", 0, _sha(0), rec_label="A_1")), art)
    c.eq(Lm.run_state(P8b, "launchintent", 0)[0], "FAIL", "a run record whose label is another run's")
    # meta: DONE must come AFTER the last pull, carry the registered text, and need all four pulls
    c.eq(Lm.log_meta(Lm.parse_boxlog(lg(runs_lines()[:-3] + [X.done_line()] + runs_lines()[-3:]), art))["done"], 0, "a DONE line BEFORE the last pull is not the chain's DONE")
    c.eq(Lm.log_meta(Lm.parse_boxlog(lg(runs_lines() + [X.done_line().replace("launch-intent-w 1.58926", "launch-intent-w 1.5")]), art))["done"], 0, "a DONE line of another dose")
    c.eq(Lm.log_meta(Lm.parse_boxlog(lg(runs_lines()), art))["done"], 0, "no DONE line")
    three = runs_lines()[:-3]
    m3 = Lm.log_meta(Lm.parse_boxlog(lg(three + [X.done_line()]), art))
    c.eq((m3["done"], m3["pending"], m3["pulled"]), (0, 1, 3), "a DONE line with only three pulls is not DONE")
    c.eq(Lm.log_meta(Lm.parse_boxlog(lg(runs_lines() + ["ZZV8FMINI-STOP something", X.done_line()]), art))["stops"], 1, "STOP lines are counted")
    return c.fails


def t_log(Lm, Tm, cid):
    c = Chk()
    art = LOGART()
    a, s = "launchintent", 0
    if cid == "log-cfail":
        P = Lm.parse_boxlog(lg(X.log_run(art, a, s, _sha(0)) + [f"ZZLAUNCHINTENT-CONTROL-FAIL {L.run_name(a, s)} K-SHARE"]), art)
        c.eq(Lm.run_state(P, a, s)[0], "FAIL", "a CONTROL-FAIL line after a PASSing run's lines")
        P = Lm.parse_boxlog(lg(X.log_run(art, a, s, _sha(0)) + [f"ZZLAUNCHINTENT-CONTROL-FAIL {L.run_name('launchintentrep', 1)} K-SHARE"]), art)
        c.eq(Lm.next_ready(P, {L.run_name(a, s)})[0], "FAIL", "a CONTROL-FAIL for a run that has no pull line fails next_ready")
    elif cid == "log-path":
        P = Lm.parse_boxlog(lg(X.log_run(art, a, s, _sha(0), path="E:/elsewhere/M2b_s0_launchintent")), art)
        c.eq(Lm.run_state(P, a, s)[0], "FAIL", "the box-2 chain pulled to another directory than this chain reads")
        P = Lm.parse_boxlog(lg(X.log_run(art, a, s, _sha(0), path=art.replace("/", "\\") + "\\M2b_s0_launchintent")), art)
        c.eq(Lm.run_state(P, a, s)[0], "PASS", "a backslash spelling of the same directory is the same directory")
    elif cid == "log-label":
        P = Lm.parse_boxlog(lg(X.log_run(art, a, s, _sha(0), label="A_1", rec=None, ksh=None)), art)
        c.eq(Lm.run_state(P, a, s)[0], "FAIL", "A_1's label on A_0's pull line (arm / seed confusion)")
        P = Lm.parse_boxlog(lg(X.log_run(art, a, s, _sha(0), control="REP_s0")), art)
        c.eq(Lm.run_state(P, a, s)[0], "FAIL", "A_0 against REP instead of DAG-OACC")
    elif cid == "log-stop-after":
        P = Lm.parse_boxlog(lg(X.log_run(art, a, s, _sha(0), rec=None, ksh=None) + ["ZZV8FMINI-STOP the box-2 chain stopped"]), art)
        c.eq(Lm.run_state(P, a, s)[0], "FAIL", "a STOP after the pull and before both controls printed PASS")
        P = Lm.parse_boxlog(lg(["ZZV8FMINI-STOP an earlier failed attempt"] + X.log_run(art, a, s, _sha(0), rec=None, ksh=None)), art)
        c.eq(Lm.run_state(P, a, s)[0], "AWAITING-CONTROLS", "a STOP BEFORE the pull line is another attempt's")
        P = Lm.parse_boxlog(lg(X.log_run(art, a, s, _sha(0)) + ["ZZV8FMINI-STOP something later"]), art)
        c.eq(Lm.run_state(P, a, s)[0], "PASS", "a later STOP does not undo a run whose controls both printed PASS")
        P = Lm.parse_boxlog(lg(X.log_run(art, a, s, _sha(0), rec=None, ksh=None) + ["ZZDISK-FLOOR-STOP no space"]), art)
        c.eq(Lm.run_state(P, a, s)[0], "FAIL", "a disk-floor STOP counts as a STOP")
    return c.fails


# ================================================================================================================================ A. guards, world, templates, commits, small lib functions
def t_guards(Lm, Tm):
    c = Chk()
    os.makedirs(PYW, exist_ok=True)
    hf = f"{PYW}/hook_ram_mb.txt"
    df = f"{PYW}/hook_disk_gb.txt"
    for f in (hf, df):
        if os.path.exists(f):
            os.remove(f)
    with envset(LIT1_WORLD=PYW, LIT1_TEST_RAM_MB="100", LIT1_TEST_DISK_GB=None):
        rc, txt = Lm.guards(4096, None, 0)
        c.eq(rc, 1, "RAM 100 MB < 4096")
        c.has(txt, "RAM available 100 MB", "the RAM message")
    with envset(LIT1_WORLD=PYW, LIT1_TEST_RAM_MB="4096"):
        c.eq(Lm.guards(4096, None, 0)[0], 0, "RAM exactly at the minimum holds")
    with envset(LIT1_WORLD=PYW, LIT1_TEST_RAM_MB="4095"):
        c.eq(Lm.guards(4096, None, 0)[0], 1, "RAM one MB under the minimum is LOW")
    with envset(LIT1_WORLD=PYW, LIT1_TEST_DISK_GB="1"):
        rc, txt = Lm.guards(0, "F:/", 5)
        c.eq(rc, 1, "disk 1 GB < 5")
        c.has(txt, "free 1.0 GB", "the disk message")
    with envset(LIT1_WORLD=PYW, LIT1_TEST_DISK_GB="5"):
        c.eq(Lm.guards(0, "F:/", 5)[0], 0, "disk exactly at the minimum holds")
    open(hf, "w").write("50")
    with envset(LIT1_WORLD=PYW, LIT1_TEST_RAM_MB="100000"):
        rc, txt = Lm.guards(4096, None, 0)
        c.eq(rc, 1, "the hook FILE wins over the env hook")
        c.has(txt, "RAM available 50 MB", "the file hook's value")
    os.remove(hf)
    with envset(LIT1_WORLD=None, LIT1_TEST_RAM_MB="1"):
        rc, txt = Lm.guards(1, None, 0)
        c.no(txt, "ram=1MB", "without a world the RAM hook is IGNORED (the real RAM is read)")
    with envset(LIT1_WORLD=HERE, LIT1_TEST_RAM_MB="1"):
        rc, txt = Lm.guards(1, None, 0)
        c.no(txt, "ram=1MB", "a LIT1_WORLD outside the scratch root does not enable the hook")
    open(hf, "w").write("7")
    with envset(LIT1_WORLD=None):
        c.no(Lm.guards(1, None, 0)[1], "ram=7MB", "the hook FILE is ignored without a world too")
    os.remove(hf)
    return c.fails


def t_world(Lm, Tm):
    c = Chk()
    old = Lm.SCRATCH
    scr, sib = f"{PYW}/scr", f"{PYW}/scr_sibling"
    for d in (f"{scr}/inside", f"{sib}/inside", f"{PYW}/outside"):
        os.makedirs(d, exist_ok=True)
    Lm.SCRATCH = scr
    try:
        with envset(LIT1_WORLD=f"{scr}/inside"):
            c.ok(Lm.world_ok(), "a directory inside the scratch root is a world")
        with envset(LIT1_WORLD=f"{sib}/inside"):
            c.ok(not Lm.world_ok(), "a SIBLING whose name merely starts with the scratch root's is not a world")
        with envset(LIT1_WORLD=f"{scr}/inside/../../scr_sibling/inside"):
            c.ok(not Lm.world_ok(), "a .. escape is resolved before the test")
        with envset(LIT1_WORLD=f"{PYW}/outside"):
            c.ok(not Lm.world_ok(), "outside")
        with envset(LIT1_WORLD=f"{scr}/absent"):
            c.ok(not Lm.world_ok(), "a world that is not a directory")
        with envset(LIT1_WORLD=scr):
            c.ok(not Lm.world_ok(), "the scratch root itself is not a world")
        with envset(LIT1_WORLD=None):
            c.ok(not Lm.world_ok(), "no variable, no world")
    finally:
        Lm.SCRATCH = old
    return c.fails


def t_render(Lm, Tm):
    c = Chk()
    ar = L.AR
    banked = open(f"{ar}/v8f-dagoacc-t1/argv/DAGOACC_s0_m10_causal_heads.argv", encoding="utf-8").read().replace("\r", "").split("\n")
    banked = [x for x in banked if x != ""]
    got = Lm.render("causal", ckpt=f"{ar}/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt", tag="DAGOACC_s0_m10_causal_heads", tree=Lm.BASE_TREE, out=f"{ar}/v8f-dagoacc-t1")
    c.eq(got, banked, "rollout_causal.tmpl rendered with DAG-OACC s0's parameters == its banked argv, token for token")
    lg_ = Lm.render("logged", ckpt="C:/x/ck.pt", tag="T", tree=Lm.BASE_TREE, out="F:/o")
    ca_ = Lm.render("causal", ckpt="C:/x/ck.pt", tag="T", tree=Lm.BASE_TREE, out="F:/o")
    diff = [(a, b) for a, b in zip(ca_, lg_) if a != b]
    c.eq(diff, [("causal", "logged")], "the logged lane differs from the causal lane in the --limit value ONLY")
    c.eq(len(ca_), len(lg_), "same token count")
    t0 = Lm.render("t0fresh", ckpt="C:/x/ck.pt", tag="T", tree=Lm.BASE_TREE, out="F:/o")
    c.ok("--ages" in t0 and t0[t0.index("--ages") + 1] == "fresh", "the fresh-age T0 eval")
    c.ok(t0[t0.index("--out-dir") + 1] == "F:/o/t0fresh", "T0 goes to <out>/t0fresh")
    c.ok(t0[1].endswith("/stack/scripts/v8f_mini_eval.py"), "the T0 eval is v8f_mini_eval.py")
    c.ok(ca_[1].endswith("/stack/scripts/v8f_t1_eval.py"), "the rollout is v8f_t1_eval.py")
    for flag, val in (("--replan-steps", "10"), ("--nav", "deadreckon"), ("--scene", "logged"), ("--schedule", "periodic"), ("--device", "cpu")):
        c.ok(flag in ca_ and ca_[ca_.index(flag) + 1] == val, f"{flag} {val}")
    c.ok("--dump-ticks" in ca_ and "--dump-heads" in ca_, "dumps")
    rd = Lm.render("reader", out="F:/o", reader="F:/r.py", prereg="F:/p.md", prereg_sha="ab" * 32)
    c.ok(rd[rd.index("--manifest") + 1] == "F:/o/runs.json" and rd[rd.index("--out") + 1] == "F:/o/read/LAUNCH_INTENT_READ.json", "reader paths")
    c.ok(rd[rd.index("--prereg-sha") + 1] == "ab" * 32, "reader prereg sha")
    old = Lm.tmpl_tokens
    Lm.tmpl_tokens = lambda kind: ["a", "@ZZ_UNKNOWN@"]
    try:
        try:
            Lm.render("causal", ckpt="x", tag="y", out="z")
            c.fails.append("an unresolved placeholder was not refused")
        except SystemExit:
            pass
    finally:
        Lm.tmpl_tokens = old
    c.eq(Lm.fmt(["a b", "x", "A_rep,0", "a&b", "E:/p/q-r.pt"]), "'a b' x A_rep,0 'a&b' E:/p/q-r.pt", "fmt quoting")
    c.eq(Lm.cmd_sha(["a", "b"]), hashlib.sha256(b"a\x00b\x00").hexdigest(), "cmd_sha is sha256 of token + NUL")
    for k in ("rollout_causal.tmpl", "rollout_logged.tmpl", "t0fresh.tmpl", "reader.tmpl"):
        raw = open(f"{HERE}/argv/{k}", "rb").read()
        c.ok(b"\r" not in raw, f"{k} is LF only")
    return c.fails


def t_small(Lm, Tm):
    c = Chk()
    c.ok(abs(Lm.share_of(30790, 191311, 1.58926) - 0.331843) < 1e-5, "share_of(30790, 191311, W) is the trainer's launch_weight_share")
    c.ok(abs(Lm.share_of(1, 2, 1.0) - 2.0 / 3.0) < 1e-12, "share_of by hand")
    c.eq(Lm.share_of(0, 0, 1.0), None, "no eligible rows")
    c.eq(Lm.npath("E:\\a\\b/"), "E:/a/b", "npath backslashes")
    c.eq(Lm.npath("/e/a/b"), "E:/a/b", "npath msys")
    c.eq(Lm.tail2("E:/x/M2b_s0_launchintent/ckpt_last.pt"), "M2b_s0_launchintent/ckpt_last.pt", "tail2")
    c.eq(Lm.flags_of(["--a", "1", "--b", "--a", "2", "3"]), {"--a": [["1"], ["2", "3"]], "--b": [[]]}, "flags_of")
    c.eq(Lm.one(Lm.flags_of(["--a", "1"]), "--a"), "1", "one")
    c.eq(Lm.one(Lm.flags_of(["--a", "1", "--a", "1"]), "--a"), None, "one refuses a repeated flag")
    c.eq(Lm.fnum("nan"), None, "fnum refuses NaN")
    c.eq(Lm.fnum("1.5"), 1.5, "fnum")
    c.eq(Lm.t1_tag("launchintentrep", 1, "causal"), "LIREP_s1_m10_causal_heads", "t1_tag")
    c.eq(Lm.eval_tag("launchintent", 0), "M2b-launchintent_s0", "eval_tag")
    c.eq(Lm.t0f_tag("launchintent", 0), "M2b-launchintent_s0-fresh", "t0f_tag")
    c.eq(len(Lm.sums_names("x")), 8, "the accepted SHA256SUMS set is 8 names")
    return c.fails


def t_commits(Lm, Tm):
    c = Chk()
    art = f"{PYW}/art_commits"
    shutil.rmtree(art, ignore_errors=True)

    def put(commit, rsha, psha, runs):
        for (a, s) in runs:
            X.mk_run(art, a, s, commit=commit, rsha=rsha, psha=psha)
    c.eq(Lm.commits_of(art)["ok"], False, "no token at all")
    put(L.LAUNCH_COMMIT, "ab" * 32, "cd" * 32, L.RUNS)
    o = Lm.commits_of(art)
    c.ok(o["ok"] and o["commits"] == [L.LAUNCH_COMMIT] and o["n_tokens"] == 4, "four agreeing tokens")
    X.mk_run(art, "launchintent", 1, commit="e" * 40, rsha="ab" * 32, psha="cd" * 32)
    c.ok(not Lm.commits_of(art)["ok"], "a token binding another commit")
    X.mk_run(art, "launchintent", 1, commit=L.LAUNCH_COMMIT, rsha="12" * 32, psha="cd" * 32)
    c.ok(not Lm.commits_of(art)["ok"], "a token binding another reader sha256")
    X.mk_run(art, "launchintent", 1, commit=L.LAUNCH_COMMIT, rsha="ab" * 32, psha="34" * 32)
    c.ok(not Lm.commits_of(art)["ok"], "a token binding another PREREG sha256")
    X.mk_run(art, "launchintent", 1, commit="abc", rsha="ab" * 32, psha="cd" * 32)
    shutil.rmtree(f"{art}/M2b_s0_launchintent")
    shutil.rmtree(f"{art}/M2b_s0_launchintentrep")
    shutil.rmtree(f"{art}/M2b_s1_launchintentrep")
    c.ok(not Lm.commits_of(art)["ok"], "a commit that is not 40 hex")
    X.mk_run(art, "launchintent", 1)
    shutil.rmtree(art)
    X.mk_run(art, "launchintent", 0, commit=L.LAUNCH_COMMIT, rsha="ab" * 32, psha="cd" * 32)
    c.ok(Lm.commits_of(art)["ok"], "ONE token that is well formed suffices (the others have not landed yet)")
    return c.fails


# ================================================================================================================================ A. the tool tree, the AST proofs, rolloutcode
MODEL_A = "from dataclasses import dataclass\n\n\n@dataclass\nclass V8FMiniConfig:\n    a: int = 1\n\n\ndef forward_pass(x):\n    return x + 1\n"
MODEL_B = MODEL_A.replace("    a: int = 1\n", "    a: int = 1\n    launch_intent_w: float = 0.0\n") + "\n\ndef launch_weight(x):\n    return x\n"
DATA_A = "class EgoPerturb:\n    def go(self, x):\n        return x\n\n\ndef other(x):\n    return x\n"


def sha_s(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def mk_tree(root, model_src):
    shutil.rmtree(root, ignore_errors=True)
    files = {"stack/scripts/v8f_t1_eval.py": "t1 = 1\n", "stack/scripts/v8f_mini_eval.py": "t0 = 1\n", "stack/tanitad/__init__.py": "", "stack/tanitad/models/__init__.py": "", L.MODEL_PATH: model_src,
             "stack/tanitad/other.py": "o = 1\n", "stack/tanitad/cfg.yaml": "k: 1\n", "taniteval/taniteval/__init__.py": ""}
    for p, s in files.items():
        os.makedirs(os.path.dirname(f"{root}/{p}"), exist_ok=True)
        open(f"{root}/{p}", "w", newline="\n").write(s)
    return root


def t_proof(Lm, Tm, cid=None):
    c = Chk()
    ok = Tm.modelproof(MODEL_A, MODEL_B)
    c.ok(ok["ok"] and "launch_weight" in ok["added"], f"the launch overlay shape is accepted: {ok['problems']}")
    rm = Tm.modelproof(MODEL_A, MODEL_B.replace("def forward_pass(x):\n    return x + 1\n", ""))
    c.ok(not rm["ok"] and rm["removed"] == ["forward_pass"], "[proof-removed] a removed definition is refused")
    ad = Tm.modelproof(MODEL_A, MODEL_B + "\n\ndef helper(x):\n    return x\n")
    c.ok(not ad["ok"] and any("helper" in p for p in ad["problems"]), "[proof-added] an added definition that does not mention the lever is refused")
    ln = Tm.modelproof(MODEL_A, MODEL_B.replace("return x + 1", "return x + 2"))
    c.ok(not ln["ok"] and any("forward_pass" in p for p in ln["problems"]), "[proof-lines] a changed line that does not mention the lever is refused")
    lk = Tm.modelproof(MODEL_A, MODEL_B.replace("return x + 1", "return x + 1  # launch"))
    c.ok(lk["ok"], "a comment-only change is not a change of the inference path")
    cm = Tm.modelproof(MODEL_A, MODEL_B.replace("    return x + 1", "    # explain\n    launch_flag = 1\n    return x + 1"))
    c.ok(cm["ok"], f"a comment line beside a lever line in a changed function is skipped: {cm['problems']}")
    ws = Tm.modelproof(MODEL_A, MODEL_B.replace("return x + 1", "return  x + 1"))
    c.ok(ws["ok"], "a whitespace-only change is not a change of the inference path")
    # the data-module rule: a whole class that is the perturbation's own is accepted without the line test
    data_b = DATA_A.replace("return x\n\n\ndef other", "return x * 2\n\n\ndef other")
    d1 = Tm.src_diff(DATA_A, data_b, r"EGO_PERTURB", name_ok=r"EgoPerturb(\.|$)")
    c.ok(d1["ok"], f"a change INSIDE the EgoPerturb class is accepted by name: {d1['problems']}")
    d2 = Tm.src_diff(DATA_A, DATA_A.replace("def other(x):\n    return x", "def other(x):\n    return x + 1"), r"EGO_PERTURB", name_ok=r"EgoPerturb(\.|$)")
    c.ok(not d2["ok"], "a change outside the perturbation class that does not mention it is refused")
    return c.fails


def t_tree_clause(Lm, Tm, cid):
    c = Chk()
    root = f"{PYW}/tree_{cid}"
    base = mk_tree(f"{root}/base", MODEL_A)
    tree = mk_tree(f"{root}/tree", MODEL_B)
    bd = Tm.tree_digest(base)
    kw = {"model_sha": sha_s(MODEL_B), "base_digest": bd}
    ok, info = Tm.treecheck(tree, base, None, **kw)
    c.ok(ok, f"the clean synthetic tree verifies: {info['problems']}")
    if cid == "tree-set":
        open(f"{tree}/stack/tanitad/zzz.py", "w").write("z = 1\n")
    elif cid == "tree-onefile":
        open(f"{tree}/stack/tanitad/other.py", "w").write("o = 2\n")
    elif cid == "tree-extras":
        open(f"{tree}/stack/tanitad/cfg.yaml", "w").write("k: 2\n")
    elif cid == "tree-overlay":
        kw["model_sha"] = "0" * 64
    elif cid == "tree-base-digest":
        kw["base_digest"] = "0" * 64
    elif cid in ("proof-removed", "proof-added", "proof-lines"):
        bad = {"proof-removed": MODEL_B.replace("def forward_pass(x):\n    return x + 1\n", ""), "proof-added": MODEL_B + "\n\ndef helper(x):\n    return x\n",
               "proof-lines": MODEL_B.replace("return x + 1", "return x + 2")}[cid]
        open(f"{tree}/{L.MODEL_PATH}", "w", newline="\n").write(bad)
        kw["model_sha"] = sha_s(bad)
    ok, info = Tm.treecheck(tree, base, None, **kw)
    c.ok(not ok, f"[{cid}] the violating synthetic tree was NOT refused")
    if cid == "tree-extras":
        os.remove(f"{tree}/stack/tanitad/cfg.yaml")
        ok2, info2 = Tm.treecheck(tree, base, None, **kw)
        c.ok(not ok2 and any("extras" in p for p in info2["problems"]), "[tree-extras] a MISSING extra is refused too")
    if cid == "tree-set":
        c.ok(any("digest set differs" in p for p in info["problems"]), "the reason names the digest set")
    # cfg reach: the tree's config must accept every field the trainer writes
    c.eq(Tm.cfg_reach(MODEL_A, MODEL_B)["missing"], ["launch_intent_w"], "a pinned-style config rejects the launch field")
    c.eq(Tm.cfg_reach(MODEL_B, MODEL_B)["missing"], [], "the overlay config accepts it")
    return c.fails


def t_tree_real(Lm, Tm):
    c = Chk()
    tree = "F:/TanitAD-scratch/launch_intent_t1/tree_li"
    ok, info = Tm.treecheck(tree, Lm.BASE_TREE, MIRROR)
    c.ok(ok, f"the real LAUNCH-INTENT tree verifies: {info['problems']}")
    c.eq(info.get("differs_from_base"), [L.MODEL_PATH], "differs from the pinned base in exactly the model")
    c.eq(info.get("base_digest"), L.BASE_DIGEST, "the pinned base digest reproduces")
    c.eq(Tm.tree_digest(Lm.BASE_TREE), L.BASE_DIGEST, "tree_digest reproduces the digest CTRL and REP were rolled out with")
    mb = Tm.blob(MIRROR, L.LAUNCH_COMMIT, L.MODEL_PATH)
    c.ok(mb is not None and Tm.sha_b(mb) == L.LAUNCH_MODEL_SHA, "the launch model blob is the pinned one")
    c.ok(Tm.blob(MIRROR, L.LAUNCH_COMMIT, "stack/NO_SUCH_FILE.py") is None, "an absent blob is None (the control of the blob probe)")
    real_blob, dest = Tm.blob, f"{PYW}/tb_absent"
    shutil.rmtree(dest, ignore_errors=True)
    Tm.blob = lambda m, cm, p: b"not the launch model"
    try:
        Tm.treebuild(dest, Lm.BASE_TREE, MIRROR)
        c.fails.append("treebuild accepted a model blob that is not the pinned launch model")
    except RuntimeError as ex:
        c.has(str(ex), "launch commit's model blob", "treebuild refuses the wrong blob BEFORE copying anything (the sha256 pin, not the later tree check)")
    finally:
        Tm.blob = real_blob
    c.ok(not os.path.exists(dest), "a refused build leaves nothing behind")
    pf = Tm.modelproof(open(f"{Lm.BASE_TREE}/{L.MODEL_PATH}", encoding="utf-8").read(), mb.decode("utf-8"))
    c.ok(pf["ok"] and pf["added"] and not pf["removed"], f"modelproof(real pinned, real launch): {pf['problems']}")
    c.ok(any("launch" in a.lower() for a in pf["added"]), "the added definitions are the lever's")
    return c.fails


def t_rolloutcode(Lm, Tm):
    c = Chk()
    r = Tm.rolloutcode(MIRROR, L.LAUNCH_COMMIT, Lm.BASE_TREE)
    c.ok(r["ok"], f"rolloutcode at the pinned flag commit: {r['problems']}")
    c.eq(r["n"], 45, "the runtime set is 45 files")
    c.ok(L.MODEL_PATH in r["differs"], "the model differs (it is the overlay)")
    real_blob = Tm.blob

    def with_blob(fn):
        Tm.blob = fn
        try:
            return Tm.rolloutcode(MIRROR, L.LAUNCH_COMMIT, Lm.BASE_TREE)
        finally:
            Tm.blob = real_blob
    other = "stack/tanitad/models/hjepa.py"
    r2 = with_blob(lambda m, cm, p: (real_blob(m, cm, p) + b"\n# x\n") if p == other else real_blob(m, cm, p))
    c.ok(not r2["ok"] and any(other in p for p in r2["problems"]), "[rolloutcode] a runtime file that differs from the pinned tree is a finding")
    mod = Tm.blob(MIRROR, L.LAUNCH_COMMIT, L.MODEL_PATH).decode("utf-8")
    bad_mod = (mod + "\n\ndef zz_unrelated(x):\n    return x\n").encode("utf-8")
    r3 = with_blob(lambda m, cm, p: bad_mod if p == L.MODEL_PATH else real_blob(m, cm, p))
    c.ok(not r3["ok"] and any(p.startswith("model:") for p in r3["problems"]), "[rolloutcode] a model change that does not mention the lever is a finding")
    dat = Tm.blob(MIRROR, L.LAUNCH_COMMIT, L.DATA_PATH)
    if dat is not None and dat != open(f"{Lm.BASE_TREE}/{L.DATA_PATH}", "rb").read():
        bad_dat = dat + b"\n\ndef zz_unrelated(x):\n    return x\n"
        r4 = with_blob(lambda m, cm, p: bad_dat if p == L.DATA_PATH else real_blob(m, cm, p))
        c.ok(not r4["ok"] and any(p.startswith("data:") for p in r4["problems"]), "[rolloutcode] a data-module change beyond the perturbation is a finding")
    r5 = with_blob(lambda m, cm, p: None if p == other else real_blob(m, cm, p))
    c.ok(not r5["ok"], "[rolloutcode] a runtime file absent at the launch commit is a finding")
    return c.fails


# --- K-BUILD on a fake object store (no git write: git() and blob() are replaced by dictionary lookups)
KB_BASE, KB_MID, KB_FLAG = "b" * 40, "c" * 40, "e" * 40
KB_TRAINER = b'parser.add_argument("--launch-intent-w", type=float, default=0.0)\n'
KB_MODEL = b"class C:\n    launch_intent_w: float = 0.0\n"


def kb_store(variant=None):
    st = {}
    for cm, tr, da, mo in ((KB_BASE, b"t0", b"d0", b"m0"), (KB_MID, b"t1", b"d1", b"m1"), (KB_FLAG, KB_TRAINER, b"d1", KB_MODEL)):
        st[(cm, L.TRAINER_PATH)], st[(cm, L.DATA_PATH)], st[(cm, L.MODEL_PATH)] = tr, da, mo
    raw = T.RAWPK
    st[(KB_FLAG, f"{raw}/diff.patch")] = b"--- a/t\n+++ b/t\n-t1\n+x\n"
    st[(KB_FLAG, f"{raw}/logs/final_green.log")] = b"collected\n36 passed in 5.0s\n"
    st[(KB_FLAG, f"{raw}/mutation.txt")] = b"SUMMARY: 30/30 mutants killed; survivors: none\n"
    st[(KB_FLAG, "stack/tests/test_v8f_launch_intent.py")] = b"def test_flag_off_reproduces_the_unmodified_tip_bit_for_bit():\n    pass\n"
    st[(KB_FLAG, "stack/tests/launch_intent_golden.json")] = b"{}"
    return st


def kb_run(Tm, st, compose=None, launch=KB_FLAG, diff_text="--- a/f\n+++ b/f\n@@ -1 +1 @@\n-t1\n+x\n"):
    real_git, real_blob = Tm.git, Tm.blob

    def fgit(mirror, *a, text=False, check=True):
        if a and a[0] == "diff":
            return diff_text
        return ""
    Tm.git, Tm.blob = fgit, (lambda m, cm, p: st.get((cm, p)))
    try:
        return Tm.kbuild("fake", launch, compose_record=compose, base_commit=KB_BASE, li_parent=KB_MID, flag_commit=KB_FLAG)
    finally:
        Tm.git, Tm.blob = real_git, real_blob


def kb_compose(path, bad=False, hunk_status="PASS", hunk_default_off=True):
    sh = lambda b: hashlib.sha256(b).hexdigest()      # noqa: E731
    rec = {"launch_commit": KB_MID, "files": {L.TRAINER_PATH: {"sha256_launch": sh(b"t1"), "sha256_base": sh(b"t0")}, L.DATA_PATH: {"sha256_launch": sh(b"d1"), "sha256_base": sh(b"d0")},
                                              L.MODEL_PATH: {"sha256_launch": sh(b"m1"), "sha256_base": sh(b"m0")}},
           "changed": [{"file": L.DATA_PATH, "hunk": "@@ -1 +1 @@", "behind_flag": "--ego-perturb-with-dagger", "default_off": hunk_default_off, "bit_identity_test": hunk_status}]}
    if bad:
        rec["files"][L.TRAINER_PATH]["sha256_launch"] = H64
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(rec, open(path, "w"))
    return path


def t_kbuild_clause(Lm, Tm, cid):
    c = Chk()
    comp = kb_compose(f"{PYW}/kb_compose.json")
    rec0 = kb_run(Tm, kb_store(), comp)
    c.ok(rec0["changed"] and all(e["bit_identity_test"] == "PASS" for e in rec0["changed"]), f"the clean fake store verifies in both steps: {[e['bit_identity_test'][:80] for e in rec0['changed']]}")
    c.ok(any(e["step"].startswith("LI_PARENT->launch") for e in rec0["changed"]), "step 2 has hunks")
    st, kw, diff = kb_store(), {"compose": comp, "launch": KB_FLAG}, "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-t1\n+x\n"
    raw = T.RAWPK
    if cid == "kb-compose-bytes":
        kw["compose"] = kb_compose(f"{PYW}/kb_compose_bad.json", bad=True)
    elif cid == "kb-launch-bytes":
        st[("f" * 40, L.TRAINER_PATH)] = KB_TRAINER + b"#"
        for p in (L.DATA_PATH, L.MODEL_PATH):
            st[("f" * 40, p)] = st[(KB_FLAG, p)]
        kw["launch"] = "f" * 40
    elif cid == "kb-diffpatch":
        st[(KB_FLAG, f"{raw}/diff.patch")] = b"--- a/t\n+++ b/t\n-t1\n+y\n"
    elif cid == "kb-green":
        st[(KB_FLAG, f"{raw}/logs/final_green.log")] = b"35 passed in 5.0s\n"
    elif cid == "kb-mutation":
        st[(KB_FLAG, f"{raw}/mutation.txt")] = b"SUMMARY: 29/29 mutants killed; survivors: none\n"
    elif cid == "kb-flagoff-test":
        st[(KB_FLAG, "stack/tests/test_v8f_launch_intent.py")] = b"def test_something_else():\n    pass\n"
    elif cid == "kb-golden":
        del st[(KB_FLAG, "stack/tests/launch_intent_golden.json")]
    elif cid == "kb-default-off":
        st[(KB_FLAG, L.TRAINER_PATH)] = b'parser.add_argument("--launch-intent-w", type=float, default=1.0)\n'
    rec = kb_run(Tm, st, kw["compose"], kw["launch"], diff)
    step2 = [e for e in rec["changed"] if e["step"].startswith("LI_PARENT->launch")]
    comp_e = [e for e in rec["changed"] if e["step"].startswith("base->LI_PARENT")]
    if cid == "kb-compose-bytes":
        c.ok(comp_e and all(e["bit_identity_test"].startswith("UNVERIFIED") for e in comp_e), f"[{cid}] the compose step must be UNVERIFIED")
    else:
        c.ok(step2 and all(e["bit_identity_test"].startswith("UNVERIFIED") for e in step2), f"[{cid}] step 2 must be UNVERIFIED, got {[e['bit_identity_test'][:60] for e in step2]}")
    return c.fails


def t_kbuild_misc(Lm, Tm):
    """The non-clause arithmetic of the green log and the mutation summary (each tested at the boundary)."""
    c = Chk()
    comp = kb_compose(f"{PYW}/kb_compose.json")
    raw = T.RAWPK

    def verdict(**repl):
        st = kb_store()
        for k, v in repl.items():
            st[(KB_FLAG, {"green": f"{raw}/logs/final_green.log", "mut": f"{raw}/mutation.txt"}[k])] = v
        rec = kb_run(Tm, st, comp)
        return [e["bit_identity_test"][:6] for e in rec["changed"] if e["step"].startswith("LI_PARENT->launch")]
    c.eq(set(verdict()), {"PASS"}, "baseline")
    c.eq(set(verdict(green=b"36 passed, 1 failed in 5s\n")), {"UNVERI"}, "a failure in the green log")
    c.eq(set(verdict(green=b"10 passed in 5s\n")), {"UNVERI"}, "10 passed is not >= 36")
    c.eq(set(verdict(green=b"36 passed in 5s\n1 error in x\n")), {"UNVERI"}, "an error line in the green log")
    c.eq(set(verdict(green=b"no summary at all\n")), {"UNVERI"}, "no 'passed' count")
    c.eq(set(verdict(mut=b"SUMMARY: 28/30 mutants killed; survivors: 2\n")), {"UNVERI"}, "survivors")
    c.eq(set(verdict(mut=b"SUMMARY: 5/5 mutants killed; survivors: none\n")), {"UNVERI"}, "fewer than 30 mutants")
    c.eq(set(verdict(mut=b"SUMMARY: 29/30 mutants killed; survivors: none\n")), {"UNVERI"}, "n killed != n total")
    c.eq(set(verdict(mut=b"")), {"UNVERI"}, "an empty mutation file")
    for what, kw in (("a compose hunk whose bit-identity test is not PASS", {"hunk_status": "FAIL"}), ("a compose hunk that is not default-off", {"hunk_default_off": False})):
        rec = kb_run(Tm, kb_store(), kb_compose(f"{PYW}/kb_compose_h.json", **kw))
        ce = [e for e in rec["changed"] if e["step"].startswith("base->LI_PARENT")]
        c.ok(ce and all(e["bit_identity_test"].startswith("UNVERIFIED") for e in ce), f"{what} is UNVERIFIED in the record")
    return c.fails


def t_kbuild_real(Lm, Tm):
    c = Chk()
    rec = Tm.kbuild(MIRROR, L.LAUNCH_COMMIT)
    c.eq(rec["launch_commit"], L.LAUNCH_COMMIT, "launch commit")
    steps = {}
    for e in rec["changed"]:
        steps.setdefault(e["step"].split(" ")[0], []).append(e["bit_identity_test"][:6])
    c.ok(steps.get("base->LI_PARENT") and set(steps["base->LI_PARENT"]) == {"PASS"}, f"the compose step (inherited, bound by blob sha): {steps}")
    c.ok(steps.get("LI_PARENT->launch") and set(steps["LI_PARENT->launch"]) == {"PASS"}, f"the launch-intent step (bound to the tested bytes): {steps}")
    ev = rec["evidence"]
    c.ok(ev.get("diff_patch") and ev.get("green_log", {}).get("passed", 0) >= 36 and ev.get("mutation", {}).get("killed", 0) >= 30 and ev.get("flag_off_test"), f"the evidence block is complete: {sorted(ev)}")
    c.eq(L.sha256_file(T.COMPOSE_RECORD), ev["compose_record"]["sha256"], "the inherited record is bound by sha256")
    return c.fails


# ================================================================================================================================ A. static scans of the shipped files
FORBID = [(r"(?<![A-Za-z0-9_])ssh(?![A-Za-z0-9_])", "a remote-shell command"), (r"(?<![A-Za-z0-9_])scp(?![A-Za-z0-9_])", "a remote-copy command"), (r"(?<![A-Za-z0-9_])sftp(?![A-Za-z0-9_])", "sftp"),
          (r"(?<![A-Za-z0-9_])rsync(?![A-Za-z0-9_])", "rsync"), (r"tanitad-" + "box2", "the box-2 alias"), (r"Keys\.txt", "the token file"), (r"hf_[A-Za-z0-9]{8,}", "a token"),
          (r"git\s+(?:-c\s+\S+\s+|-C\s+\S+\s+)*(add|commit|push|checkout|reset|merge|rebase|tag|stash|clean|rm|mv|apply|am|cherry-pick|update-ref|write-tree|commit-tree|hash-object|update-index|fetch|pull|clone|init|config)(?![-A-Za-z0-9_])", "a git write")]
GIT_OK = {"cat-file", "diff", "rev-parse", "merge-base", "show", "log"}


def t_static(Lm, Tm):
    c = Chk()
    for f in SHIPPED:
        p = f"{HERE}/{f}"
        if not os.path.exists(p):
            c.fails.append(f"{f}: shipped file is missing")
            continue
        raw = open(p, "rb").read()
        txt = raw.decode("utf-8")
        if f.endswith(".cmd"):
            c.ok(raw.count(b"\r\n") == raw.count(b"\n") and b"\r\n" in raw, f"{f}: a .cmd file is CRLF only")
        else:
            c.ok(b"\r" not in raw, f"{f}: LF only")
        c.ok(all(ord(ch) >= 32 or ch in "\n\r\t" for ch in txt), f"{f}: no control character other than tab / newline (a corrupted heredoc escape)")
        for rx, what in FORBID:
            for m in re.finditer(rx, txt):
                line = txt[:m.start()].count("\n") + 1
                c.fails.append(f"{f}:{line}: {what} ({m.group(0)[:40]!r}): the shipped chain must never contain it")
                break
        if f.endswith(".py"):
            try:
                import ast
                ast.parse(txt)
            except SyntaxError as ex:
                c.fails.append(f"{f}: does not parse: {ex}")
    for m in re.finditer(r'git\(mirror, "([a-z-]+)"', open(f"{HERE}/lit1_tree.py", encoding="utf-8").read()):
        c.ok(m.group(1) in GIT_OK, f"lit1_tree.py calls git {m.group(1)}: only read-only plumbing is allowed {sorted(GIT_OK)}")
    for m in re.finditer(r'git -c safe\.directory=\S+ -C "\$MIRROR" ([a-z-]+)', open(f"{HERE}/lit1_chain.sh", encoding="utf-8").read()):
        c.ok(m.group(1) in GIT_OK, f"lit1_chain.sh calls git {m.group(1)}")
    for f in ("lit1_chain.sh", "wake_lit1.sh"):
        rc, out = S.run(f'bash -n "{HERE}/{f}"', env={"LIT1_WORLD": PYW})
        c.eq(rc, 0, f"bash -n {f}: {out[:200]}")
    cmd = open(f"{HERE}/lit1_chain.cmd", "rb").read().decode("utf-8") if os.path.exists(f"{HERE}/lit1_chain.cmd") else ""
    c.has(cmd, "lit1_chain.sh", "the launcher runs lit1_chain.sh")
    c.has(cmd, "chain_v8f_launchintent", "the launcher names box 2's log (rem)")
    return c.fails


# ================================================================================================================================ the registry of python tests and mutants
LANDED_COMMIT = "77234276db2cc18498ee6ef639865cf82944c680"
LANDED_PREREG_SHA = "469bac1f3918b31b875e7c06b637980f6acb194a69ded3294c180aec95b1f65a"
LANDED_READER_SHA = "5e950dcff2993653dffbe9b4f01ea17eb4d7b1721b7c41a669e0dc43c559fc35"
READER_PATH = f"{L.AR}/dagland/launch_intent_v2/read/launch_intent_read.py"
READER_REPO = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-28-launch-intent-v2/read/launch_intent_read.py"


def t_landed(Lm, Tm):
    """The coordinator's pins: the landed PREREG (469bac1f...) and the landed reader (5e950dcf...) are the bytes this chain reads."""
    import subprocess
    c = Chk()
    p = f"{HERE}/landed/PREREG_V8F_LAUNCH_INTENT.md"
    raw = open(p, "rb").read()
    c.eq(hashlib.sha256(raw).hexdigest(), LANDED_PREREG_SHA, "landed/PREREG_V8F_LAUNCH_INTENT.md is the landed PREREG (coordinator's pin)")
    c.ok(b"\r" not in raw, "the PREREG copy is LF only (as in the git object)")
    blob = Tm.git("C:/Users/Admin/tanitad-push", "cat-file", "blob", f"{LANDED_COMMIT}:Project Steering/PREREG_V8F_LAUNCH_INTENT.md")
    c.eq(hashlib.sha256(blob).hexdigest(), LANDED_PREREG_SHA, "the git object at the landed commit hashes to the pin")
    c.eq(blob, raw, "the copy is byte-identical to the git object")
    c.eq(Lm.sha256_file(READER_PATH), LANDED_READER_SHA, "the reader at the chain's default path has the landed reader's sha256")
    rb = Tm.git("C:/Users/Admin/tanitad-push", "cat-file", "blob", f"{LANDED_COMMIT}:{READER_REPO}")
    c.eq(hashlib.sha256(rb).hexdigest(), LANDED_READER_SHA, "the landed reader's git object hashes to the pin")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    ok = subprocess.run([Lm.PY, "-I", READER_PATH, "--static-check", "--prereg", p, "--prereg-sha", LANDED_PREREG_SHA], capture_output=True, env=env, stdin=subprocess.DEVNULL)
    c.ok(ok.returncode == 0 and b"LAUNCH-INTENT-READ-STATIC-OK" in ok.stdout, f"reader --static-check with the landed pin: rc {ok.returncode}")
    bad = subprocess.run([Lm.PY, "-I", READER_PATH, "--static-check", "--prereg", p, "--prereg-sha", "0" * 64], capture_output=True, env=env, stdin=subprocess.DEVNULL)
    c.ok(bad.returncode == 2 and b"STATIC-REFUSED" in bad.stdout, f"reader --static-check with a wrong pin must REFUSE (rc 2): rc {bad.returncode}")
    chain = open(f"{HERE}/lit1_chain.sh", encoding="utf-8").read()
    c.has(chain, LANDED_PREREG_SHA, "the chain pins the landed PREREG sha256")
    c.has(chain, LANDED_READER_SHA, "the chain pins the landed reader sha256")
    c.has(chain, "$H/landed/PREREG_V8F_LAUNCH_INTENT.md", "the chain's default PREREG is the landed copy")
    return c.fails


def py_tests():
    """name -> fn(Lm, Tm)"""
    t = {"clause_positive": t_clause_positive, "pull_misc": t_pull_misc, "log_misc": t_log_misc, "guards": t_guards, "world": t_world, "render": t_render, "small": t_small, "commits": t_commits,
         "proof": t_proof, "tree_real": t_tree_real, "rolloutcode": t_rolloutcode, "kbuild_misc": t_kbuild_misc, "kbuild_real": t_kbuild_real, "static": t_static, "landed": t_landed}
    for cid in CASES:
        t[f"clause:{cid}"] = (lambda Lm, Tm, cid=cid: t_clause(Lm, Tm, cid))
    for cid in PULL_IDS:
        t[f"pull:{cid}"] = (lambda Lm, Tm, cid=cid: t_pull(Lm, Tm, cid))
    for cid in LOG_IDS:
        t[f"log:{cid}"] = (lambda Lm, Tm, cid=cid: t_log(Lm, Tm, cid))
    for cid in TREE_IDS:
        t[f"tree:{cid}"] = (lambda Lm, Tm, cid=cid: t_tree_clause(Lm, Tm, cid))
    for cid in KB_IDS:
        t[f"kb:{cid}"] = (lambda Lm, Tm, cid=cid: t_kbuild_clause(Lm, Tm, cid))
    return t


def clause_mutants():
    """(mutant id, set of clause ids switched off, test name that must then FAIL)"""
    out = [(f"on:{cid}", {cid}, f"clause:{cid}") for cid in CASES if cid not in ("sum-mask",)]
    out.append(("on:sum-mask", {"sum-mask"}, "clause:sum-mask"))
    out += [(f"on:{cid}", {cid}, f"pull:{cid}") for cid in PULL_IDS]
    out += [(f"on:{cid}", {cid}, f"log:{cid}") for cid in LOG_IDS]
    out += [("on:guard-ram", {"guard-ram"}, "guards"), ("on:guard-disk", {"guard-disk"}, "guards")]
    out += [(f"on:{cid}", {cid}, f"tree:{cid}") for cid in TREE_IDS]
    out += [(f"on:{cid}", {cid}, f"kb:{cid}") for cid in KB_IDS]
    return out


R_BS = "\\"
TEXT_MUTANTS_PY = [      # (id, file, old, new, test)
    ("L-pull-order", "lit1_lib.py", "return [(a, s) for (_l, a, s) in sorted(have)]", "return [(a, s) for (_l, a, s) in have]", "log_misc"),
    ("L-complete-skip", "lit1_lib.py", "        if run_name(a, s) in complete:\n            continue\n", "", "log_misc"),
    ("L-stop-before-pull", "lit1_lib.py", 'if i > s["line"]]', 'if i >= 0]', "log:log-stop-after"),
    ("L-done-after-last", "lit1_lib.py", "any(i > last and DONE_RE.match(t) for (i, t) in P[\"dones\"])", "any(DONE_RE.match(t) for (i, t) in P[\"dones\"])", "log_misc"),
    ("L-done-all-pulled", "lit1_lib.py", '"done": int(len(pulled) == len(RUNS) and any(', '"done": int(any(', "log_misc"),
    ("L-stray-rec", "lit1_lib.py", 'if cur is None or g != cur:\n                problems.append', 'if False:\n                problems.append', "log_misc"),
    ("L-stray-ksh", "lit1_lib.py", 'if cur is None or sec[cur]["label"] != m.group(2):', 'if False:', "log_misc"),
    ("L-sums-newline", "lit1_lib.py", 'if not raw.endswith("' + R_BS + 'n"):', "if False:", "pull_misc"),
    ("L-sums-dup", "lit1_lib.py", "if m.group(2) in listed:", "if False:", "pull_misc"),
    ("L-sums-empty", "lit1_lib.py", "or os.path.getsize(p) == 0:", "or False:", "pull_misc"),
    ("L-sums-malformed", "lit1_lib.py", "if not m:\n            return 3, \"WRONG\"", "if False:\n            return 3, \"WRONG\"", "pull_misc"),
    ("L-share-formula", "lit1_lib.py", "return ((1.0 + w) * n) / ((1.0 + w) * n + (e - n))", "return (w * n) / (w * n + (e - n))", "small"),
    ("L-world-prefix", "lit1_lib.py", 'return r.startswith(s + "/") and os.path.isdir(w)', 'return r.startswith(s) and os.path.isdir(w)', "world"),
    ("L-world-isdir", "lit1_lib.py", 'and os.path.isdir(w)', '', "world"),
    ("L-guard-ram-eq", "lit1_lib.py", 'if ram < ram_min_mb and on("guard-ram"):', 'if ram <= ram_min_mb and on("guard-ram"):', "guards"),
    ("L-guard-disk-eq", "lit1_lib.py", 'if free < disk_min_gb and on("guard-disk"):', 'if free <= disk_min_gb and on("guard-disk"):', "guards"),
    ("L-hook-world", "lit1_lib.py", "if not world_ok():\n            return None", "if False:\n            return None", "guards"),
    ("L-hook-file-first", "lit1_lib.py", "if os.path.isfile(fp):\n            return open(fp).read().strip()", "if False:\n            return open(fp).read().strip()", "guards"),
    ("L-fmt-comma", "lit1_lib.py", 'r"[^A-Za-z0-9_./:=,@%+-]"', 'r"[^A-Za-z0-9_./:=@%+-]"', "render"),
    ("L-fmt-never", "lit1_lib.py", "o.append(f\"'{a}'\" if re.search(", "o.append(f\"{a}\" if re.search(", "render"),
    ("L-cmdsha-sep", "lit1_lib.py", 'a.encode() + b"' + R_BS + '0"', 'a.encode() + b"' + R_BS + 'n"', "render"),
    ("L-render-unresolved", "lit1_lib.py", 'if re.search(r"@[A-Z0-9_]+@", t):', "if False:", "render"),
    ("L-tmpl-comment", "lit1_lib.py", 'if line and not line.startswith("#"):', "if line:", "render"),
    ("L-commits-one", "lit1_lib.py", "len(com) == 1", "len(com) >= 1", "commits"),
    ("L-commits-reader", "lit1_lib.py", "len(rd) == 1", "len(rd) >= 1", "commits"),
    ("L-commits-prereg", "lit1_lib.py", "len(pr) == 1", "len(pr) >= 1", "commits"),
    ("L-commits-hex", "lit1_lib.py", 're.fullmatch(r"[0-9a-f]{40}", next(iter(com)))', "True", "commits"),
    ("L-fnum-nan", "lit1_lib.py", "return v if math.isfinite(v) else None", "return v", "small"),
    ("L-one-repeat", "lit1_lib.py", "if occ and len(occ) == 1 and len(occ[0]) == 1 else None", "if occ and len(occ[0]) == 1 else None", "small"),
    ("L-npath", "lit1_lib.py", 'p = str(p).replace("' + R_BS + R_BS + '", "/").rstrip("/")', 'p = str(p).rstrip("/")', "small"),
    ("L-t0-tag", "lit1_lib.py", 'return f"{eval_tag(arm, seed)}-fresh"', 'return f"{eval_tag(arm, seed)}"', "small"),
    ("L-sums-names", "lit1_lib.py", '"selfcheck.json", "summary.json", "argv.json", "train_log.jsonl", "launch_gate_token.json"]', '"selfcheck.json", "summary.json", "argv.json", "train_log.jsonl"]', "small"),
    ("L-label-check", "lit1_lib.py", 'sec[cur]["problems"].append(f"run-record label {m.group(2)}', 'pass  # sec[cur]["problems"].append(f"run-record label {m.group(2)}', "log_misc"),
    ("T-name-ok", "lit1_tree.py", "if nok and nok.match(k):\n            continue", "if False:\n            continue", "proof"),
    ("T-comment-line", "lit1_tree.py", 'if body and not body.startswith("#") and', "if body and", "proof"),
    ("T-rc-model", "lit1_tree.py", 'if not pf["ok"]:\n                res["problems"].append("model: "', 'if False:\n                res["problems"].append("model: "', "rolloutcode"),
    ("T-rc-else", "lit1_tree.py", 'res["problems"].append(f"{p} differs between the pinned tree and the launch commit")', "pass", "rolloutcode"),
    ("T-rc-absent", "lit1_tree.py", "if a:\n                res[\"problems\"].append(f\"{p} is absent at the launch commit", "if False:\n                res[\"problems\"].append(f\"{p} is absent at the launch commit", "rolloutcode"),
    ("T-rc-data", "lit1_tree.py", 'if not pf["ok"]:\n                res["problems"].append("data: "', 'if False:\n                res["problems"].append("data: "', "rolloutcode"),
    ("T-kb-green-n", "lit1_tree.py", "int(m[-1]) >= 36", "int(m[-1]) >= 0", "kbuild_misc"),
    ("T-kb-green-fail", "lit1_tree.py", 'and not re.search(r"' + R_BS + 'b' + R_BS + 'd+ (failed|error)", gtxt)', "", "kbuild_misc"),
    ("T-kb-mut-n", "lit1_tree.py", "int(mm.group(2)) >= 30", "int(mm.group(2)) >= 0", "kbuild_misc"),
    ("T-kb-mut-eq", "lit1_tree.py", "mm.group(1) == mm.group(2) and", "", "kbuild_misc"),
    ("T-kb-compose-pass", "lit1_tree.py", 'h.get("bit_identity_test") == "PASS" and h.get("default_off") is True', "True", "kbuild_misc"),
    ("T-treecheck-extras", "lit1_tree.py", "for p in extras(base) if not os.path.isfile(os.path.join(tree, p)) or", "for p in extras(base) if", "tree:tree-extras"),
    ("T-digest-line", "lit1_tree.py", 'f"{L.sha256_file(os.path.join(root, p))} *{p}' + R_BS + 'n"', 'f"{L.sha256_file(os.path.join(root, p))}  {p}' + R_BS + 'n"', "tree_real"),
    ("T-treebuild-sha", "lit1_tree.py", "if mb is None or sha_b(mb) != L.LAUNCH_MODEL_SHA:", "if mb is None:", "tree_real"),
]


if __name__ == "__main__":
    import lit1_selftest_run
    sys.exit(lit1_selftest_run.main())
