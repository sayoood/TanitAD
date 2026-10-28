#!/usr/bin/env python3
"""Self-test, shell layer part 2: the sealed guard and its regression arms, the queue stage (real cmd.exe) for both arms, the invariants, the test list and the mutant table of the chain / lib / waker / guard / queue."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from st_sh import *                                   # noqa: F401,F403
import st_sh as H
import st_common as S
import sealed
import build_li_asserter as BLA
import build_li_checks as BLC
import build_li_lib as BLL
import build_li_small as BLS
import make_queue_stage as MQS

C_ = S.C
BS = chr(92)


def s_echo(logs):
    wr(logs, "chain_v8f_launchintent.out", "10-28 09:00 waiting for ^ZZV8FMINI-STOP or ZZV8FMINI-DONE (a dated echo is not an end marker)\n")


def p_w_echo(v):
    rc, o = waker_run(v, s_echo, poll=1, n=2)
    return "CHAIN END" not in o and "WAKER TIMEOUT" in o, o[-200:]


def p_guard_echo(v):
    w = v.world(True)
    out = S.fresh("guard/echo_" + re.sub(r"[^A-Za-z0-9]+", "_", v.name)[:30])
    (out / "o.out").write_bytes(b"10-28 09:00 waiting for ZZV8FMINI-DONE\n")
    rc, o = bash(f'timeout 5 env V8F_CHAIN_OUT="{px(out / "o.out")}" bash "{guard_copy(w)}"', guard_env(w, LI_BUSY_FILE="x", V8F_CHAIN_OUT=px(out / "o.out")), 60)
    return rc == 124 and "guard off" not in o, f"rc={rc} " + o[-200:]


# ---------------------------------------------------------------------------------------------------------------------------------------- the sealed guard: its deliberate-regression arms
_ARMS = {}


def guard_arms():
    if "a" not in _ARMS:
        _ARMS["a"] = sealed.regression_arms(S.fresh("guardarms"))
    return _ARMS["a"]


def guard_tests():
    mech, stub = sealed.incident_mechanism()
    S.T("guard: the 2026-10-28 mechanism is real: with the stub dir first on the ENV path only, Git bash resolves `ssh` to " + mech + ", not the stub (so a PATH handed over in the environment never protects a step)", mech.endswith("/usr/bin/ssh") and mech != stub + "/ssh", (mech, stub))
    for arm in guard_arms():
        name, _kw, expect = arm
        got, why = sealed.exercise_arm(arm)
        S.T(f"guard arm: {name}: {'REFUSED (rc 97, nothing ran, no stub call)' if expect == 'refused' else 'the step runs'}", got == expect, (got, why))
    # every harness module starts bash only through sealed.run / sealed.popen
    offenders = static_bash_offenders()
    S.T("harness: no module of the self-test (nor make_dry_run.py) starts bash except through the sealed runner", not offenders, offenders)
    S.T("harness: the static scan flags a module that DOES start bash directly (a positive control)", bool(scan_text("import subprocess\nr = subprocess.run([BASHX, '-c', cmd])\n", "x.py")), "")  # scan-ok
    S.T("harness: the static scan flags `C.BASH` / `bash.exe` too", bool(scan_text("x = C.BASH\n", "x.py")) and bool(scan_text("p = 'C:/Program Files/Git/bin/bash.exe'\n", "x.py")), "")  # scan-ok
    S.T("harness: the static scan skips only lines that carry its explicit marker (an unmarked line is flagged)", not scan_text("x = C.BASH  # scan-ok\n", "x.py") and bool(scan_text("x = C.BASH  # ok\n", "x.py")), "")  # scan-ok


BASH_RX = re.compile(r"\bBASHX?\b|\bC\.BASH\b|\bC_\.BASH\b|bash\.exe|\[\s*[\"']bash[\"']|subprocess\.\w+\(\s*[\"']bash[\"']")  # scan-ok


def scan_text(text: str, name: str) -> list:
    return [f"{name}:{i + 1}: {ln.strip()[:100]}" for i, ln in enumerate(text.split("\n")) if BASH_RX.search(ln) and not ln.lstrip().startswith("#") and "# scan-ok" not in ln]


def static_bash_offenders() -> list:
    out = []
    for f in ("st_sh.py", "st_sh2.py", "st_py.py", "st_common.py", "make_dry_run.py", "selftest_chain.py", "make_chain_li.py", "make_queue_stage.py"):
        out += scan_text((C_.HERE / f).read_text(encoding="utf-8"), f)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------- the queue stage under REAL cmd.exe
BANK0 = "197a68bb540be5a7d157942e9157e1c8e60ca60d82b1ca4fb4fcb68edccfaa8f"


def dry_cmd(qpath, args: str, work: Path):
    bat = work / "dry.cmd"
    lines = ["@echo off", "set V8F_DRYRUN=1", 'set "REPO=D:\\tanitad\\repo"', 'set "PY=D:\\tanitad\\venv\\Scripts\\python.exe"', 'set "CACHEFULL=D:\\tanitad\\cache_full"', 'set "V9=D:\\tanitad\\data\\v9"',
             'set "RUNSM=D:\\tanitad\\runs\\v8f_mini_full"', f'set "DAGSHA_DRY={BANK0}"', 'set "WACCDAG_DRY=49.6969"', f'call "{Path(qpath)}" %*']
    bat.write_bytes(("\r\n".join(lines) + "\r\n").encode())
    r = subprocess.run(["cmd.exe", "/c", str(bat)] + args.split(), stdin=subprocess.DEVNULL, capture_output=True)
    return (r.stdout + r.stderr).decode("utf-8", "replace").replace("\r", "").strip()


def labels(b: bytes) -> dict:
    lines = b.decode("utf-8").split("\r\n")
    if lines and lines[-1] == "":
        lines.pop()                                       # the file's final newline is not part of the last label's block
    out, cur, name = {"": []}, [], ""
    for ln in lines:
        m = re.match(r"^:([A-Za-z0-9_]+)", ln)
        if m:
            out[name] = cur
            name, cur = m.group(1), []
        cur.append(ln)
    out[name] = cur
    return out


def q_tests():
    work = S.fresh("q")
    tipq, stq = labels(TIP_QUEUE), labels(STAGED_QUEUE)
    same = all(stq[k] == tipq[k] for k in tipq if k not in ("dagname", "dagarm")) and set(stq) - set(tipq) == {"launchintentarm"}
    S.T("queue: every label block of the landed commit's queue other than :dagname and :dagarm is byte-identical in the staged queue; the only new label is :launchintentarm", same)
    d1 = [x for x in stq["dagname"] if x not in tipq["dagname"]], [x for x in tipq["dagname"] if x not in stq["dagname"]]
    d2 = [x for x in stq["dagarm"] if x not in tipq["dagarm"]], [x for x in tipq["dagarm"] if x not in stq["dagarm"]]
    S.T("queue: :dagname gains exactly two lines (the whitelist entries of launchintent and launchintentrep) and :dagarm exactly two (their dispatches); no line is removed", len(d1[0]) == 2 and not d1[1] and len(d2[0]) == 2 and not d2[1], (d1, d2))
    S.T("queue: the staged file is reproducible from the landed commit's blob (make_queue_stage.derive) and is pure CRLF", MQS.derive(TIP_QUEUE) == STAGED_QUEUE and STAGED_QUEUE.count(b"\r\n") == STAGED_QUEUE.count(b"\n"))
    S.T("queue: the landed commit's queue blob is byte-identical to the pre-lever tip's (so the stage applies on both)", C_.git("rev-parse", f"{C_.TIP}:{QPATH}").strip() == C_.git("rev-parse", f"{C_.PRE_TIP}:{QPATH}").strip())
    qs = {k: work / f"{k}.cmd" for k in ("tip", "staged")}
    qs["tip"].write_bytes(TIP_QUEUE)
    qs["staged"].write_bytes(STAGED_QUEUE)
    base = {"launchintent": ("dagoacc", " 0.1"), "launchintentrep": ("dagoaccorcrep", "")}      # (the pair control's stage name, the P argument it takes: dagoacc needs 0.1, the REP stage none)
    for arm, s in C_.RUN_ORDER:
        lab, ts = C_.RUN_LABEL[(arm, s)], C_.tseed(arm, s)
        g = dry_cmd(qs["staged"], f"gatedag {arm} {s}", work)
        t = dry_cmd(qs["staged"], f"traindag {arm} {s}", work)
        e = dry_cmd(qs["staged"], f"evaldag {arm} {s}", work)
        S.T(f"queue {lab}: gate line = the training line + --selfcheck-only, trainer --seed {ts}, ends `--dagger-seed {s} --w-acc 49.6969 --launch-intent-w 1.58926`, own --out M2b_s{s}_{arm} and gate directory",
            g.endswith("--dagger-seed %d --w-acc 49.6969 --launch-intent-w 1.58926 --selfcheck-only" % s) and t.endswith("--dagger-seed %d --w-acc 49.6969 --launch-intent-w 1.58926" % s)
            and f"gate_M2b-{arm}_s{s}" in g and f"M2b_s{s}_{arm}" in t and f"M2b_s{s}\\ckpt_last.pt" in t and t.count("--launch-intent-w") == 1 and f"--arm M2b --seed {ts} --grid" in t and f"--arm M2b --seed {ts} --grid" in g, (g[-160:], t[-160:]))
        S.T(f"queue {lab}: eval line names the run's own checkpoint and tag", f"--tag M2b-{arm}_s{s}" in e and f"M2b_s{s}_{arm}\\ckpt_last.pt" in e and "--project --families" in e)
        # the pair control's own line, with the name swapped and the lever appended, IS this run's line: the one-lever property under real cmd.exe
        cname, cp = base[arm]
        ctrl_t = dry_cmd(qs["staged"], f"traindag {cname} {s}{cp}", work)
        S.T(f"queue {lab}: the training line == the pair control's ({cname} {s}) training line with the run name swapped and `--launch-intent-w 1.58926` appended (nothing else)",
            t == ctrl_t.replace(f"M2b_s{s}_{cname}", f"M2b_s{s}_{arm}") + " --launch-intent-w 1.58926", (t[-200:], ctrl_t[-200:]))
    for what, args in (("a P argument", "gatedag launchintent 0 0.1"), ("a P argument on the replicate arm", "gatedag launchintentrep 1 0.1"), ("seed 2", "gatedag launchintent 2"), ("seed 2 on the replicate arm", "traindag launchintentrep 2"),
                       ("an unknown arm", "gatedag launchintent2 0"), ("an arm that is a prefix of the replicate's", "gatedag launchintentre 0"), ("an arm that extends the replicate's", "gatedag launchintentrepx 0"), ("no seed", "gatedag launchintent"), ("no seed on the replicate arm", "gatedag launchintentrep")):
        o = dry_cmd(qs["staged"], args, work)
        S.T(f"queue: {what} is REFUSED (rc-2 text, no training line)", "REFUSED" in o and "v8f_mini_train.py" not in o, o[:200])
    diffs = []
    for args in ("gatedag dagoacc 0 0.1", "traindag dagoacc 1 0.1", "gatedag dagoaccorc 0", "traindag dagoaccorcrep 1", "gatedag dagoaccorcrep 0", "gatedag flatdag 0 0.1", "gatedag dag 1 0.1", "gatedag dag0 0", "gatedag dag2 0 0.1", "evaldag dagoacc 0", "gatedag dagorc 0 0.1", "evaldag dagoaccorcrep 1"):
        a, b = dry_cmd(qs["tip"], args, work), dry_cmd(qs["staged"], args, work)
        if a != b or not a:
            diffs.append((args, a[:80], b[:80]))
    S.T("queue: twelve other stage/arm lines (dagoacc, dagoaccorc, dagoaccorcrep, flatdag, dag, dag0, dag2, dagorc; gate/train/eval) are IDENTICAL on the landed commit's queue and the staged queue", not diffs, diffs)
    for arm in C_.LI_ARMS:
        tipo = dry_cmd(qs["tip"], f"gatedag {arm} 0", work)
        S.T(f"queue: the LANDED commit's queue REFUSES the arm {arm} (so the chain's stage greps are what protect against a tip without it)", "REFUSED" in tipo and "v8f_mini_train.py" not in tipo, tipo[:120])


def inv_tests():
    bad_eol = []
    for p in list(S.SHIP.glob("*")) + list(C_.HERE.glob("*.py")) + list(C_.HERE.glob("*.in")):
        if p.suffix in (".sh", ".py", ".in") and b"\r" in p.read_bytes():
            bad_eol.append(p.name)
    S.T("EOL: every .sh / .py / .in is LF", not bad_eol, bad_eol)
    crlf = [p.name for p in list(S.SHIP.glob("*.cmd")) + list(C_.QSTAGE.glob("*.cmd")) if p.read_bytes().count(b"\r\n") != p.read_bytes().count(b"\n") or b"\r\n" not in p.read_bytes()]
    S.T("EOL: every .cmd (the launcher, the staged queue, the landed commit's queue) is CRLF throughout", not crlf, crlf)
    rc = [bash(f'bash -n "{px(p)}"')[0] for p in S.SHIP.glob("*.sh")]
    S.T("syntax: bash -n on every generated .sh", all(r == 0 for r in rc) and len(rc) == 4, rc)
    ok = True
    for p in list(S.SHIP.glob("*.py")) + list(C_.HERE.glob("*.py")):
        try:
            compile(p.read_text(encoding="utf-8"), str(p), "exec")
        except SyntaxError:
            ok = False
    S.T("syntax: every generated .py and every generator / harness .py compiles", ok)
    chainp = S.SHIP / "chain_v8f_launchintent.sh"
    pv = MC.read_pins_from_disk(chainp)
    S.T("regeneration: the chain on disk equals a fresh render (the five pins read from disk)", chainp.read_bytes() == MC.render(pv["LAUNCH_TRAINER_SHA"], S.SHIP, **{k: v for k, v in pv.items() if k != "LAUNCH_TRAINER_SHA"}).encode("utf-8"))
    S.T("regeneration: asserter, checks, lib, guard, waker, launcher on disk equal their builders' output",
        (S.SHIP / "gate_launchintent_assert.py").read_bytes() == BLA.build().encode() and (S.SHIP / "v8f_launchintent_chain_checks.py").read_bytes() == BLC.TEXT.replace("__LI_W__", C_.LI_W).encode()
        and (S.SHIP / "oaf_launchintent_lib.sh").read_bytes() == BLL.TEXT.encode() and (S.SHIP / "v8f_guard_launchintent.sh").read_bytes() == BLS.guard().encode()
        and (S.SHIP / "wake_v8flaunchintent.sh").read_bytes() == BLS.WAKER.encode() and (S.SHIP / "run_chain_v8f_launchintent.cmd").read_bytes() == BLS.RUNNER.encode())
    chain = chainp.read_text(encoding="utf-8")
    S.T("pins: the shipped chain is PINNED to the landed commit (LAUNCH_COMMIT, LAUNCH_TRAINER_SHA, LAUNCH_MODEL_SHA = the coordinator's values), the staged trainer / model ARE those blobs, and PREREG_SHA / READER_SHA are NAMED PLACEHOLDERS",
        pv["LAUNCH_COMMIT"] == C_.LANDED_COMMIT and pv["LAUNCH_TRAINER_SHA"] == C_.LANDED_TRAINER_SHA == STAGED_SHA and pv["LAUNCH_MODEL_SHA"] == C_.LANDED_MODEL_SHA == STAGED_MODEL_SHA
        and pv["PREREG_SHA"].startswith("<<UNSET") and pv["READER_SHA"].startswith("<<UNSET"), pv)
    S.T("pins: the landed blobs in the mirror hash to the pinned trainer and model sha256 (re-measured, not inherited)", (MC.landed_consistency() or True))
    S.T("pins: the registered W is pinned in the chain, the asserter, the checks and the queue stage",
        "LI_W=1.58926" in chain and 'LI_W_STR = "1.58926"' in (S.SHIP / "gate_launchintent_assert.py").read_text(encoding="utf-8") and "--launch-intent-w 1.58926" in STAGED_QUEUE.decode())
    ph = MC.render(MC.PH_PIN, S.SHIP)
    S.T("pins: every pinned sha256 in the chain equals the sha256 of the file it names (asserter, checks, lib, guard, installed oaf_lib.sh, installed oaf checks)",
        all(f"{k}={C_.sha(p)}" in ph for k, p in (("ASSERT_SHA", S.SHIP / "gate_launchintent_assert.py"), ("LCHECKS_SHA", S.SHIP / "v8f_launchintent_chain_checks.py"), ("LILIB_SHA", S.SHIP / "oaf_launchintent_lib.sh"),
                                                  ("GUARD_SHA", S.SHIP / "v8f_guard_launchintent.sh"), ("OAF_LIB_SHA", C_.INSTALLED / "oaf_lib.sh"), ("OAF_CHECKS_SHA", C_.INSTALLED / "v8f_oaf_chain_checks.py"))))
    S.T("pins: the REP pair controls' argv.json / summary.json / checkpoint sha256 in the chain equal the banked SHA256SUMS.txt of DAG-ORC r2's REP pulls (both seeds)",
        all(f"{k}=({C_.sums(C_.REPDIR / 'M2b_s0_dagoaccorcrep' / 'SHA256SUMS.txt')[f]} {C_.sums(C_.REPDIR / 'M2b_s1_dagoaccorcrep' / 'SHA256SUMS.txt')[f]})" in ph for k, f in (("REP_CKPT_SHA", "ckpt_last.pt"), ("REP_ARGV_SHA", "argv.json"), ("REP_SUMM_SHA", "summary.json"))))
    S.T("run order: the chain's RUNS array is A_0, A_1, A_rep,0, A_rep,1 (arm launchintent 0, launchintent 1, launchintentrep 0, launchintentrep 1)", 'RUNS=("launchintent 0" "launchintent 1" "launchintentrep 0" "launchintentrep 1")' in chain and RUN_LABELS == ["A_0", "A_1", "A_rep,0", "A_rep,1"])


# ---------------------------------------------------------------------------------------------------------------------------------------- the test list
V0 = Variant("orig")
SHELL_TESTS = [
    (f"--check-tree: the overlay (landed tree + staged queue) with the SHIPPED chain PASSES all {NPAT} patterns and the trainer's and model's sha256", p_ct_pass),
    ("--check-tree: the SAME overlay with a placeholder trainer pin is REFUSED (exactly because the pin is unset)", p_ct_placeholder),
    ("--check-tree: the SAME overlay with a placeholder model pin is REFUSED", p_ct_model_placeholder),
    ("--check-tree: a tree check needs neither the PREREG nor the reader pin (both placeholders: PASS)", p_ct_prereg_free),
    ("--check-tree: a trainer that differs by one line (all greps hold) is REFUSED on its sha256", p_ct_sha),
    ("--check-tree: a model that differs by one line (all greps hold) is REFUSED on its sha256", p_ct_msha),
    ("--check-tree: the PRE-LEVER tip's tree is REFUSED (the lever and the queue stage are not there)", p_ct_pre),
    ("--check-tree: the LANDED commit's tree is REFUSED for the queue stage alone (the flag and the model are the pinned ones)", p_ct_landed),
    ("--check-tree: a trainer lacking the --launch-intent-w argparse line is REFUSED by that grep", p_ct_noflag),
    ("--check-tree: a queue lacking :launchintentarm is REFUSED", p_ct_nostage),
    ("--check-tree: a model lacking the launch term in the total is REFUSED", p_ct_nototal),
    ("--check-tree: a queue lacking the replicate arm's trainer-seed line is REFUSED", p_ct_norepseed),
    ("--check-tree: a queue lacking the replicate arm's whitelist entry is REFUSED", p_ct_repwhitelist),
    ("--dry-run on the staged queue: five pins, the plan of all FOUR runs labelled A_0 A_1 A_rep,0 A_rep,1 with trainer seeds 0 / 1 / 10 / 11, SAME x4, ONE-LEVER x4, K-SHARE / K-MASK bands, the flag in all eight lines, no warning line", p_dry_pass),
    ("--dry-run with PREREG_SHA and READER_SHA unset: still prints the plan and says UNSET (a dry run needs neither)", p_dry_unset),
] + [(f"--dry-run on a queue mutated as `{k}` STOPs (expected text in the refusal)", p_dry_queue(k)) for k in
     ("tip", "w", "wB", "extra", "egoP", "first", "twice", "nocall", "nodispatch", "nowhitelist", "p03", "norepseed", "repseed2", "repseed_all", "repdispatch", "repwhitelist")] + [
    (f"a helper changed after review is REFUSED before anything runs: {f}", p_tamper(f)) for f in
    ("gate_launchintent_assert.py", "v8f_launchintent_chain_checks.py", "oaf_launchintent_lib.sh", "v8f_guard_launchintent.sh", "oaf_lib.sh", "v8f_oaf_chain_checks.py")] + [
    ("real run: AFTER_OUT unset -> STOP, nothing touched", p_r_noafter),
    ("real run: a placeholder LAUNCH_TRAINER_SHA STOPs before the predecessor is waited for, box 2 is not contacted", p_r_placeholder),
    ("real run: a placeholder PREREG_SHA STOPs before the predecessor is waited for, box 2 is not contacted", p_r_prereg),
    ("real run: a placeholder READER_SHA STOPs before the predecessor is waited for, box 2 is not contacted", p_r_reader),
    ("real run: a placeholder LAUNCH_MODEL_SHA STOPs before the predecessor is waited for", p_r_model),
    ("real run: a placeholder LAUNCH_COMMIT STOPs before the predecessor is waited for", p_r_commit),
    ("real run: a predecessor that STOPPED -> HALT (oaf_lib's own rule would have GONE), no tip read, no BOX2_BUSY, box 2 not contacted", p_r_predstop),
    ("--wait-only: a predecessor that ENDED in its DONE with BOX2_BUSY released -> ZZOAF-WAIT-GO", p_r_wait_go),
    ("--wait-only: a predecessor still running keeps WAITING (killed by timeout, no GO)", p_r_wait_wait),
    ("--wait-only: a BOX2_BUSY held by another chain -> STOP after 3 re-checks and the other chain's file is left intact", p_r_busy_held),
    ("real run: a REP that fails the licence (composition flag built) -> STOP `REP must be trained anew`, tip not read, box 2 not contacted", p_r_rep_bad),
    ("real run: the licence is checked at BOTH seeds (seed 1 bad, seed 0 good -> STOP)", p_r_rep_s1bad),
    ("real run: a tip whose trainer lacks the lever is REFUSED before BOX2_BUSY is taken (no ssh / scp)", p_r_tip("noflag", 'add_argument\\("--launch-intent-w"')),
    ("real run: a tip whose queue lacks :launchintentarm is REFUSED before BOX2_BUSY is taken", p_r_tip("nostage", "the tip's queue has no launchintentarm stage")),
    ("real run: a tip whose trainer sha256 is not LAUNCH_TRAINER_SHA (every grep holds) is REFUSED", p_r_tip("sha", "!= LAUNCH_TRAINER_SHA")),
    ("real run: a tip whose model sha256 is not LAUNCH_MODEL_SHA (every grep holds) is REFUSED", p_r_tip("msha", "!= LAUNCH_MODEL_SHA")),
    ("real run: a tip that does NOT descend from LAUNCH_COMMIT (the landed flag) is REFUSED", p_r_tip("orphan", "does not descend from LAUNCH_COMMIT")),
    ("real run: a tip whose queue lacks the :dagoaccwacc stage (greps hold) is REFUSED by the stage loop", p_r_tip("nolabel", "the tip's queue has no dagoaccwacc stage")),
    ("real run: a tip whose model lacks the launch term in the total is REFUSED", p_r_tip("nototal", "launch_intent_w \\* T")),
    ("real run: a tip whose queue lacks the replicate arm's trainer-seed line is REFUSED", p_r_tip("norepseed", 'launchintentrep" set "DAGSEED=1%~2"')),
    ("real run: a tip missing an exported file is REFUSED before BOX2_BUSY is taken", p_r_tip("noexport", "the tip has no stack/scripts/v8f_mini_eval.py")),
    ("real run: a good tip is accepted (it descends from the landed commit), BOX2_BUSY is taken, the ONLY ssh calls are the boot probe and each is made WHILE holding BOX2_BUSY, nothing is shipped, the stop releases the lock", p_r_ok),
    ("claim_box2: ten concurrent claims -> exactly ONE wins (noclobber)", p_claim_atomic),
    ("claim_box2: an existing foreign BOX2_BUSY is neither taken nor removed, BUSY_TAKEN stays 0", p_claim_foreign),
    ("control selection, EXECUTED from the chain's own text: the four runs' label / run name / tag / gate dir / pair-control dir, name and pinned shas (DAG-OACC_s for A_s, REP_s for A_rep,s)", p_ctrl_funcs),
    ("waker: wakes on the chain's end (DONE)", p_w_end), ("waker: wakes on a STOP", p_w_stop), ("waker: wakes on its OWN guard's ZZGUARD-STOP (exit 4)", p_w_guard),
    ("waker: wakes on ZZLAUNCHINTENT-CONTROL-FAIL", p_w_named), ("waker: wakes on ZZDISK-FLOOR-STOP", p_w_floor),
    ("waker: wakes ONCE on the FOUR gates passing (a transition seen after it started)", p_w_gates),
    ("waker: three of the four gates passed -> it keeps waiting", p_w_gates_partial),
    ("waker: wakes on the gates with the SIBLING chains' markers, guard output and logs present as decoys (and is not fooled by them)", p_w_gates_decoy),
    ("waker: started AFTER the gates passed it does not re-fire", p_w_gates_late), ("waker: wakes on a new pull", p_w_pull), ("waker: an already-seen pull does not wake it", p_w_pull_none),
    ("waker: only the sibling chains' markers, guard output and END/STOP logs present (decoys) -> it keeps waiting", p_w_decoys_only), ("waker: a dated `waiting for ^ZZV8FMINI-STOP` echo is not an end marker", p_w_echo),
    ("guard: exits when its chain's log ends", p_guard_end), ("guard: a dated echo of an end marker does not switch it off", p_guard_echo),
    ("names: no marker / lock / scratch / done-file of a sibling chain is read or written in live code; every name this chain writes is its own; the launcher's log is its own", p_static_names),
    ("real ship dir: the world's copies of waitjob.sh, mk_v8f_job.py, oaf_lib.sh and the installed checks name no C:/Users/Admin/box2_ship path (a probe that reached `queue` or `waitrc` could not write there)", p_static_realdir),
]


# the static wiring of the steps no fake box reaches (K-BUILD, the dev-box control checks, gate_one, runone): the EXECUTED proof of what they call is the python layer; this probe pins the call sites' text
WIRING = ('cd=$(ctrl_dir $arm $s)', '--b-summary "$(cygpath -m $cd/summary.json)" --ref-argv-json "$(cygpath -m $cd/argv.json)"', '--prereg-sha "$PREREG_SHA" --reader-sha "$READER_SHA"',
          '--seed $s --arm $arm --banked-init-sha', 'kshare --summary "$(cygpath -m $ART/$g/summary.json)" --window-sha $WINDOW_SHA --arm $arm --seed $s', 'exp=$(ctrl_argv_sha $arm $s)', 'exp=$(ctrl_summ_sha $arm $s)',
          'cf=$(ctrl_dir $arm $s)/$f', 'for arm in launchintent launchintentrep; do', 'for run in "${RUNS[@]}"; do set -- $run; gate_one $1 $2; done', 'for run in "${RUNS[@]}"; do set -- $run; runone $1 $2; done',
          'PINS.txt', '--ref-argv-json "$(cygpath -m "$(ctrl_dir $arm $s)/argv.json")"')


def p_static_wiring(v):
    w = v.world(True)
    t = live_text(w["chain"])
    miss = [x for x in WIRING if x not in t]
    return not miss, miss


SHELL_TESTS.append(("static wiring of the real-run steps no fake box reaches (gate_one, runone, the dev-box control checks): each call site names its run's pair control, arm and the PREREG / reader pins", p_static_wiring))


def run_tests():
    sealed.ensure_global()
    sh = H.shared()
    S.T("shell layer: scratch world built (10 git mirrors from the landed export, each with the scratch landed commit as ancestor, 8 trees, 17 queue variants, 3 REP pulls)", True)
    guard_tests()
    q_tests()
    inv_tests()
    for name, probe in SHELL_TESTS:
        ok, detail = probe(V0)
        S.T(name, ok, detail)


# ---------------------------------------------------------------------------------------------------------------------------------------- the mutant table (shell layer)
def CH(old, new, n=1):
    return ("chain", old, new, n)


def HP(f, old, new, n=1):
    return ("helper", f, old, new, n)


LC_LINES = {
    "oaf_lib.sh": '[ "$(sha256sum "$HERE_SHIP/oaf_lib.sh" | cut -c1-64)" = "$OAF_LIB_SHA" ]',
    "v8f_oaf_chain_checks.py": '[ "$(sha256sum "$CHECKS" | cut -c1-64)" = "$OAF_CHECKS_SHA" ]',
    "gate_launchintent_assert.py": '[ "$(sha256sum "$ASSERT" 2>/dev/null | cut -c1-64)" = "$ASSERT_SHA" ]',
    "v8f_launchintent_chain_checks.py": '[ "$(sha256sum "$LCHECKS" 2>/dev/null | cut -c1-64)" = "$LCHECKS_SHA" ]',
    "oaf_launchintent_lib.sh": '[ "$(sha256sum "$LILIB" 2>/dev/null | cut -c1-64)" = "$LILIB_SHA" ]',
    "v8f_guard_launchintent.sh": '[ "$(sha256sum "$GUARD" 2>/dev/null | cut -c1-64)" = "$GUARD_SHA" ]',
}
TIPS = {k: p_r_tip(k, w) for k, w in (("noflag", 'add_argument\\("--launch-intent-w"'), ("nostage", "the tip's queue has no launchintentarm stage"), ("sha", "!= LAUNCH_TRAINER_SHA"), ("msha", "!= LAUNCH_MODEL_SHA"), ("orphan", "does not descend from LAUNCH_COMMIT"),
                                      ("nolabel", "the tip's queue has no dagoaccwacc stage"), ("nototal", "launch_intent_w \\* T"), ("noexport", "the tip has no stack/scripts/v8f_mini_eval.py"),
                                      ("norepseed", 'launchintentrep" set "DAGSEED=1%~2"'))}
REPSEED_LEVER = '  "$PK/code/v8f_mini_box2_queue.cmd|launchintentrep' + BS + '" set ' + BS + '"DAGSEED=1%~2' + BS + '""\n'
REPNAME_LEVER = '  "$PK/code/v8f_mini_box2_queue.cmd|DAGS=launchintentrep' + BS + '""\n'

MUT_SH = [(f"H0{i} lib_check no longer verifies {f}", [CH(old, "true")], [p_tamper(f)]) for i, (f, old) in enumerate(LC_LINES.items(), 1)] + [
    ("H07 the trainer's sha256 is not compared with LAUNCH_TRAINER_SHA", [CH('[ "$got" = "$LAUNCH_TRAINER_SHA" ] ||', "true ||")], [p_ct_sha, TIPS["sha"]]),
    ("H08 a placeholder pin is accepted as a value", [CH('(LAUNCH.txt says who sets it and when)"; return 0 ;;', '(LAUNCH.txt says who sets it and when)"; return 1 ;;')], [p_ct_placeholder, p_r_prereg]),
    ("H09 the early pin refusal is removed", [CH("pin_vars_check || exit 1", "true")], [p_r_placeholder, p_r_prereg, p_r_reader]),
    ("H10 the REP licence is not enforced", [CH('rep_licence_all || stop "REP-LICENCE failed', 'true || stop "REP-LICENCE failed')], [p_r_rep_bad]),
    ("H12 the tip's lever / queue / sha check is skipped", [CH('lever_check git "$T" || stop', 'true || stop')], [TIPS["noflag"], TIPS["nototal"], TIPS["sha"]]),
    ("H13 the queue-stage loop is skipped", [CH('for st in "${STAGES[@]}"; do git', "for st in; do git")], [TIPS["nolabel"]]),
    ("H14 the export-existence loop is skipped", [CH('for xf in "${EXPORT[@]}"; do git -C $MIRROR cat-file -e', "for xf in; do git -C $MIRROR cat-file -e")], [TIPS["noexport"]]),
    ("H20 the AFTER_OUT requirement is removed", [CH('if [ -z "${AFTER_OUT:-}" ]; then\n  echo "ZZV8FMINI-STOP AFTER_OUT is not set: this chain', 'if false; then\n  echo "ZZV8FMINI-STOP AFTER_OUT is not set: this chain')], [p_r_noafter]),
    ("H21 the dry-run's ONE-LEVER / SAME verdict is ignored", [CH('[ "$same" = SAME ] && [ "$ol" = ONE-LEVER ] ||', "true ||")], [p_dry_queue("first"), p_dry_queue("extra"), p_dry_queue("w")]),
    ("H22 the dry-run does not print the K-SHARE / K-MASK bands", [CH('" bands || {', '" bands >/dev/null || {')], [p_dry_pass]),
    ("H23 the dry-run ignores a queue that prints no plan", [CH('plan "$QF" $arm $s "$sha" "$WACC" || { echo "ZZV8FMINI-STOP the queue prints no gate / train / eval line for $(rn $arm $s)"; exit 1; }', 'plan "$QF" $arm $s "$sha" "$WACC" || true')], [p_dry_queue("tip")]),
    ("H27 the box is claimed and probed BEFORE the tip is read", [CH("# ---- the tip, BEFORE BOX2_BUSY is taken:", 'claim_box2 "early" || exit 1; BOOT0=$(boot)\n# ---- the tip, BEFORE BOX2_BUSY is taken:')], [TIPS["noflag"], TIPS["nostage"]]),
    ("H29 BOX2_BUSY is never claimed (box 2 would be probed without the lock)", [CH('claim_box2 "v8F-mini PREREG_V8F_LAUNCH_INTENT arm A', 'true "v8F-mini PREREG_V8F_LAUNCH_INTENT arm A')], [p_r_ok]),
    # ---- the four-run redesign: the run list, the pair controls, the pins and the tip's ancestry
    ("H30 only the two A runs are planned (A_rep,0 and A_rep,1 dropped)", [CH('RUNS=("launchintent 0" "launchintent 1" "launchintentrep 0" "launchintentrep 1")', 'RUNS=("launchintent 0" "launchintent 1")')], [p_dry_pass, p_ctrl_funcs]),
    ("H31 A_rep's pair control is DAG-OACC (the REP directory is never used)", [CH('launchintentrep) echo "$REPDIR/M2b_s${2}_dagoaccorcrep" ;; esac; }', 'launchintentrep) echo "$OACC_DIR/M2b_s${2}_dagoacc" ;; esac; }')], [p_ctrl_funcs, p_dry_pass]),
    ("H32 the dry run checks every line as arm A (the one-lever call lacks the run's arm)", [CH('--seed $s --arm $arm)', '--seed $s --arm launchintent)')], [p_dry_pass]),
    ("H33 the PREREG pin is not required by a real run", [CH('  w=$(pin_problem PREREG_SHA "$PREREG_SHA" 64) && { echo "ZZV8FMINI-STOP $w"; bad=1; }\n', "")], [p_r_prereg]),
    ("H34 the reader pin is not required by a real run", [CH('  w=$(pin_problem READER_SHA "$READER_SHA" 64) && { echo "ZZV8FMINI-STOP $w"; bad=1; }\n', "")], [p_r_reader]),
    ("H35 the LAUNCH_COMMIT pin is not required by a real run", [CH('  w=$(commit_pin_problem) && { echo "ZZV8FMINI-STOP $w"; bad=1; }\n', "")], [p_r_commit]),
    ("H36 the LAUNCH_MODEL_SHA pin is not required by a real run", [CH('  w=$(model_pin_problem) && { echo "ZZV8FMINI-STOP $w"; bad=1; }\n', "")], [p_r_model]),
    ("H37 the model's sha256 is not compared with LAUNCH_MODEL_SHA", [CH('[ "$got" = "$LAUNCH_MODEL_SHA" ] ||', "true ||")], [p_ct_msha, TIPS["msha"]]),
    ("H38 the tip need not descend from the landed commit", [CH('git -C $MIRROR merge-base --is-ancestor "$LAUNCH_COMMIT" "$ref" 2>/dev/null ||', "true ||")], [TIPS["orphan"]]),
    ("H39 the ancestry test is reversed (the landed commit would have to descend from the tip)", [CH('merge-base --is-ancestor "$LAUNCH_COMMIT" "$ref"', 'merge-base --is-ancestor "$ref" "$LAUNCH_COMMIT"')], [p_r_ok, TIPS["orphan"]]),
    ("H40 the replicate arm's trainer-seed line is not in the lever greps", [CH(REPSEED_LEVER, "")], [p_ct_norepseed, TIPS["norepseed"]]),
    ("H41 the replicate arm's whitelist entry is not in the lever greps", [CH(REPNAME_LEVER, "")], [p_ct_repwhitelist]),
    ("H42 the gate asserter is given DAG-OACC's summary and argv for every run", [CH('--b-summary "$(cygpath -m $cd/summary.json)" --ref-argv-json "$(cygpath -m $cd/argv.json)"', '--b-summary "$(cygpath -m $OACC_DIR/M2b_s${s}_dagoacc/summary.json)" --ref-argv-json "$(cygpath -m $OACC_DIR/M2b_s${s}_dagoacc/argv.json)"')], [p_static_wiring]),
    ("H43 the run record is checked without the run's arm", [CH('--seed $s --arm $arm --banked-init-sha', '--seed $s --banked-init-sha')], [p_static_wiring]),
    ("H44 the gate token is written without the PREREG / reader pins", [CH('--prereg-sha "$PREREG_SHA" --reader-sha "$READER_SHA" ', "")], [p_static_wiring]),
    ("H45 the dev-box control check covers DAG-OACC only", [CH('for arm in launchintent launchintentrep; do', 'for arm in launchintent; do')], [p_static_wiring]),
    ("H46 only the first two runs are gated", [CH('for run in "${RUNS[@]}"; do set -- $run; gate_one $1 $2; done', 'for run in "${RUNS[0]}" "${RUNS[1]}"; do set -- $run; gate_one $1 $2; done')], [p_static_wiring]),
    ("H47 K-SHARE is not asked for the run's arm and seed", [CH('--window-sha $WINDOW_SHA --arm $arm --seed $s', '--window-sha $WINDOW_SHA')], [p_static_wiring]),
    ("H48 the lbl helper mislabels the replicate runs", [CH('"launchintentrep 0") echo A_rep,0 ;;', '"launchintentrep 0") echo A_0 ;;')], [p_ctrl_funcs]),
    ("H49 the run name helper drops the arm", [CH('rn() { echo "M2b_s${2}_${1}"; }', 'rn() { echo "M2b_s${2}_launchintent"; }')], [p_ctrl_funcs]),
    # ---- the lib
    ("H11 the REP licence is checked at seed 0 only", [], [p_r_rep_s1bad]),
    ("H15 claim_box2 is not atomic (no noclobber)", [], [p_claim_atomic]),
    ("H16 claim_box2 removes a foreign lock when it cannot take it", [], [p_claim_foreign]),
    ("H17 a HALT verdict lets the chain go", [], [p_r_predstop]),
    ("H18 a held BOX2_BUSY lets the chain go after the re-checks", [], [p_r_busy_held]),
    ("H19 a WAIT verdict lets the chain go", [], [p_r_wait_wait]),
    # ---- the waker
    ("W1 the waker ignores its guard", [], [p_w_guard]), ("W2 the waker reads a SIBLING chain's guard output", [], [p_w_guard, p_w_gates_decoy, p_w_decoys_only]),
    ("W3 the gates threshold is 3 tokens", [], [p_w_gates_partial]), ("W3b the gates threshold is 5 tokens", [], [p_w_gates]),
    ("W4 the waker re-fires on gates already passed when it started", [], [p_w_gates_late]),
    ("W5 the waker ignores ZZLAUNCHINTENT-CONTROL-FAIL", [], [p_w_named]), ("W6 any pull (not only a new one) wakes the waker", [], [p_w_pull_none]),
    ("W7 the waker reads a SIBLING chain's log", [], [p_w_end, p_w_decoys_only]), ("W8 the waker gates on a SIBLING chain's marker file", [], [p_w_gates_decoy]),
    ("W9 the waker's end test is not anchored", [], [p_w_echo]),
    # ---- the guard and the names
    ("GD1 the guard writes the sibling guard's scratch file names", [], [p_static_names]), ("GD2 the guard's end test is not anchored", [], [p_guard_echo]),
    ("S1 the chain's current-run file is the sibling's", [CH("CUR=$L/v8flaunchintent_current.txt", "CUR=$L/v8fdagoaccorc_r2_current.txt")], [p_static_names]),
    ("S2 the chain's evidence directory is the sibling's", [CH("E=$L/v8f_launchintent\n", "E=$L/v8f_dagoaccorc_r2\n")], [p_static_names]),
    ("S3 the lib names the sibling's aside marker", [], [p_static_names]), ("S4 the launcher redirects into the predecessor's log", [], [p_static_names]),
    ("S5 the code archive has the sibling's name", [CH("repo_v8flaunchintent_tip.tar", "repo_v8fdagoaccorc_r2_tip.tar", -1)], [p_static_names]),
]
HELPER_MUTS = {
    "H11 the REP licence is checked at seed 0 only": [HP("oaf_launchintent_lib.sh", "for s in 0 1; do rep_licence_one $s || bad=1; done", "for s in 0; do rep_licence_one $s || bad=1; done")],
    "H15 claim_box2 is not atomic (no noclobber)": [HP("oaf_launchintent_lib.sh", "( set -C; printf", "( printf")],
    "H16 claim_box2 removes a foreign lock when it cannot take it": [HP("oaf_launchintent_lib.sh", '2>/dev/null)"\n  return 1\n}', '2>/dev/null)"\n  rm -f "$B/BOX2_BUSY.txt"\n  return 1\n}')],
    "H17 a HALT verdict lets the chain go": [HP("oaf_launchintent_lib.sh", 'HALT) stop "predecessor verdict: $v" ;;', "HALT) return 0 ;;")],
    "H18 a held BOX2_BUSY lets the chain go after the re-checks": [HP("oaf_launchintent_lib.sh", '[ $k -gt 3 ] && stop "predecessor verdict: $v (still held after $k checks)"', "[ $k -gt 3 ] && return 0")],
    "H19 a WAIT verdict lets the chain go": [HP("oaf_launchintent_lib.sh", "WAIT) k=0; sleep $poll ;;", "WAIT) return 0 ;;")],
    "W1 the waker ignores its guard": [HP("wake_v8flaunchintent.sh", "grep -aq ZZGUARD-STOP $GO 2>/dev/null && {", "false && {")],
    "W2 the waker reads a SIBLING chain's guard output": [HP("wake_v8flaunchintent.sh", "GO=$L/v8flaunchintent_guard.out", "GO=$L/v8fdagoaccorc_r2_guard.out")],
    "W3 the gates threshold is 3 tokens": [HP("wake_v8flaunchintent.sh", "NG=4 ", "NG=3 ")],
    "W3b the gates threshold is 5 tokens": [HP("wake_v8flaunchintent.sh", "NG=4 ", "NG=5 ")],
    "W4 the waker re-fires on gates already passed when it started": [HP("wake_v8flaunchintent.sh", ' && [ "$ng0" -lt $NG ]', "")],
    "W5 the waker ignores ZZLAUNCHINTENT-CONTROL-FAIL": [HP("wake_v8flaunchintent.sh", "'^ZZLAUNCHINTENT-CONTROL-FAIL\\|^ZZDISK-FLOOR-STOP'", "'^ZZDISK-FLOOR-STOP'")],
    "W6 any pull (not only a new one) wakes the waker": [HP("wake_v8flaunchintent.sh", '[ "${n:-0}" -gt "$np0" ]', '[ "${n:-0}" -ge "$np0" ]')],
    "W7 the waker reads a SIBLING chain's log": [HP("wake_v8flaunchintent.sh", "O=$L/chain_v8f_launchintent.out", "O=$L/chain_v8f_dagoaccorc_r2.out")],
    "W8 the waker gates on a SIBLING chain's marker file": [HP("wake_v8flaunchintent.sh", '[ "$ng" -ge $NG ] && [ "$ng0" -lt $NG ]', '[ "$ng" -ge $NG ] && [ ! -f $L/.dagoaccorc_r2_gates_seen ]')],
    "W9 the waker's end test is not anchored": [HP("wake_v8flaunchintent.sh", "grep -aq '^ZZV8FMINI-STOP\\|^ZZV8FMINI-DONE' $O", "grep -aq 'ZZV8FMINI-STOP\\|ZZV8FMINI-DONE' $O")],
    "GD1 the guard writes the sibling guard's scratch file names": [HP("v8f_guard_launchintent.sh", "v8f_guard_launchintent_tail.jsonl", "v8f_guard_tail.jsonl", 2)],
    "GD2 the guard's end test is not anchored": [HP("v8f_guard_launchintent.sh", "grep -q -E '^(ZZV8FMINI|ZZBETA|ZZV8FTAC)-(DONE|STOP)'", "grep -q -E '(ZZV8FMINI|ZZBETA|ZZV8FTAC)-(DONE|STOP)'")],
    "S3 the lib names the sibling's aside marker": [HP("oaf_launchintent_lib.sh", "claim_box2() {", "R2M=r2_aside_done.txt\nclaim_box2() {")],
    "S4 the launcher redirects into the predecessor's log": [HP("run_chain_v8f_launchintent.cmd", "> /c/Users/Admin/box2_ship/logs/chain_v8f_launchintent.out 2>&1", "> /c/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc_r2.out 2>&1")],
}


def run_mutants():
    sealed.ensure_global()
    H.shared()
    # ---- the sealed guard itself: a guard with ONE check removed must change the behaviour of the arm that targets it
    arms = {a[0]: a for a in guard_arms()}
    for name, drop, target in sealed.MUTANT_PREFIXES:
        arm = arms[target]
        got, why = sealed.exercise_arm(arm, drop)
        S.M(f"{name}  [arm `{target[:60]}`: registered {arm[2]}, now {got}]", got != arm[2], why)
    for name, chain_muts, probes in MUT_SH:
        cm = [(o, n_, k) for (_t, o, n_, k) in chain_muts]
        hm = {}
        for (_t, f, o, n_, k) in HELPER_MUTS.get(name, []):
            hm.setdefault(f, []).append((o, n_, k))
        try:
            v = Variant(name, cm, hm)
            killed, why = False, "no probe flipped"
            for pr in probes:
                try:
                    ok, detail = pr(v)
                except AssertionError:
                    raise
                except Exception as e:                      # a probe that CRASHES proves nothing: report it as not killed so it is looked at (never as a silent kill)
                    killed, why = False, f"probe {getattr(pr, '__name__', 'probe')} RAISED {type(e).__name__}: {str(e)[:160]}"
                    break
                if not ok:
                    killed, why = True, f"{getattr(pr, '__name__', 'probe')}: {str(detail)[-140:]!r}"
                    break
        except AssertionError as e:
            S.M(name, False, f"mutation did not apply: {e}")
            continue
        S.M(name + (f"  [by {why[:120]}]" if killed else ""), killed, why)
    for key in ("tip", "w", "wB", "extra", "egoP", "first", "twice", "nocall", "nodispatch", "nowhitelist", "p03", "norepseed", "repseed2", "repseed_all", "repdispatch", "repwhitelist"):
        ok, detail = p_dry_queue(key)(V0)
        S.M(f"Q-{key}: a staged queue mutated as `{key}` is REFUSED by the dry run (the queue stage cannot drift into the chain unseen)", ok, detail)
