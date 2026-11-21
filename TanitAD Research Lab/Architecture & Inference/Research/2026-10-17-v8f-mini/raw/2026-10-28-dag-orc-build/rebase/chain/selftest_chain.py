#!/usr/bin/env python3
"""Self-test AND mutation proof of the DAG-ORC (DAG-OACC base) chain, asserter, run-record checks, queue stage and waiter.  CPU only; nothing touches box 2, box2_ship, git or the network.

    python selftest_chain.py [--quiet]      writes selftest_chain_result.json; rc 0 iff every test passes AND every mutant is killed

A MUTANT is a deliberate regression: a world / line / queue / chain with ONE thing wrong that the registered gate must REFUSE.  A mutant that is accepted (or refused for a different reason than the one it was built to trip) is a SURVIVOR and fails the run.
Fixtures are DAG-OACC's REAL banked gate / run records (world.py), with O-RECOVER's real ego probe grafted in for A.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "box2_ship"))
import world as W                                       # noqa: E402
import gate_dagoaccorc_assert as G                      # noqa: E402
import v8f_dagoaccorc_chain_checks as C                 # noqa: E402

SCR = Path("F:/TanitAD-scratch/dagorc_rebase_tmp/chain_selftest")
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
QUEUE = HERE / "stage" / PK / "code" / "v8f_mini_box2_queue.cmd"
TIPQ = HERE / "tip" / "queue_bf404bfe.cmd"
BASH = "C:/Program Files/Git/bin/bash.exe"
PYV = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
BOX2 = {"REPO": r"D:\tanitad\repo", "PY": r"D:\tanitad\venv\Scripts\python.exe", "CACHEFULL": r"D:\tanitad\cache_full", "V9": r"D:\tanitad\data\v9", "RUNSM": r"D:\tanitad\runs\v8f_mini_full"}
QUIET = "--quiet" in sys.argv
RESULTS, MUTANTS = [], []


def T(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    if not QUIET or not cond:
        print(("ok   " if cond else "FAIL ") + name + ("" if cond else f"  <- {str(detail)[:300]}"))


def M(name, killed, why=""):
    MUTANTS.append((name, bool(killed)))
    if not QUIET or not killed:
        print(("killed   " if killed else "SURVIVOR ") + name + ("" if killed else f"  <- {str(why)[:300]}"))


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def fresh(sub):
    d = SCR / sub
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


# ====================================================================== A. the gate asserter on four real-record worlds + the token
print("== A: gate asserter, worlds")
for arm in ("dagoaccorc", "dagoaccorcrep"):
    for s in (0, 1):
        d = W.build(fresh(f"w_{arm}_{s}"), arm, s)
        bad = quiet(W.run_check, d)
        T(f"A1 {arm} s{s}: a gate built from DAG-OACC's REAL gate record (A: + the real ego probe) PASSES every assertion", bad == [], bad)
d = W.build(fresh("tok"), "dagoaccorc", 0)
args = ["token", "--selfcheck", d["selfcheck"], "--gate-line", d["gate_line"], "--train-line", d["train_line"], "--pin-line", d["pin_line"], "--arm", "dagoaccorc", "--seed", "0", "--banked-init-sha", d["banked_init_sha"],
        "--bank-sha", d["bank_sha"], "--bank-json", d["bank_json"], "--w-acc", W.W_ACC, "--b-summary", d["b_summary"], "--ref-argv-json", d["ref_argv_json"], "--commit", "a" * 40, "--queue-sha", "b" * 64,
        "--key-sha-json", '{"f": "c"}', "--run", "M2b_s0_dagoaccorc", "--out", str(SCR / "tok" / "token.json")]
rc = quiet(G.main, args)
tok = json.load(open(SCR / "tok" / "token.json", encoding="utf-8")) if rc == 0 else {}
T("A2 token: written only after the checks pass, bound to the commit, the TRAINING argv sha256, the data sha256", rc == 0 and tok.get("binding", {}).get("commit") == "a" * 40
  and tok["binding"]["argv_sha256"] == G.argv_sha256(G.argv_of(open(d["pin_line"], encoding="utf-8").read())) and re.fullmatch(r"[0-9a-f]{64}", tok["binding"]["data_sha256"]), tok)
T("A3 token records the arm, ego p 0.3, the composition flag, w_acc, the bank sha, dagger_p, the amendment", tok.get("arm") == "dagoaccorc" and tok.get("ego_perturb_p") == 0.3 and tok.get("ego_perturb_with_dagger") is True
  and tok.get("w_acc") == 49.6969 and tok.get("dagger_p") == 0.1 and tok.get("bank_sha256") == d["bank_sha"] and "A1-2026-10-27" in tok.get("amendment", ""), tok)
bad_args = list(args)
bad_args[bad_args.index("--selfcheck") + 1] = W.build(fresh("tok2"), "dagoaccorc", 0, mutate=lambda sc: sc.update(declared_vs_built="FAIL"))["selfcheck"]
bad_args[bad_args.index("--out") + 1] = str(SCR / "tok2" / "token.json")
rc2 = quiet(G.main, bad_args)
T("A4 token: NOT written when a check fails (declared_vs_built FAIL)", rc2 == 1 and not (SCR / "tok2" / "token.json").exists(), rc2)
T("A5 seeds other than 0 / 1 are refused", quiet(G.main, [*args[:args.index("--seed") + 1], "2", *args[args.index("--seed") + 2:]]) == 2)

print("== A-mut: deliberate regressions of the gate (each must be REFUSED, and by the check built for it)")


def sc_set(path, value):
    def f(sc):
        o = sc
        for k in path[:-1]:
            o = o[k]
        o[path[-1]] = value
    return f


def sc_del(path):
    def f(sc):
        o = sc
        for k in path[:-1]:
            o = o[k]
        o.pop(path[-1], None)
    return f


def chain_mut(*fs):
    def f(sc):
        for g in fs:
            g(sc)
    return f


def argv_edit(old, new, count=1):
    """replace the value following flag ``old`` (a [flag, value] pair -> new list of tokens)"""
    def f(a):
        k = a.index(old[0])
        assert a[k + 1:k + 1 + len(old) - 1] == old[1:], (old, a[k:k + 4])
        return a[:k] + new + a[k + len(old):]
    return f


def argv_add(tokens, at=None):
    def f(a):
        return a + tokens if at is None else a[:at] + tokens + a[at:]
    return f


def argv_del(flag, n_values):
    def f(a):
        k = a.index(flag)
        return a[:k] + a[k + 1 + n_values:]
    return f


def move_flag_before(flag, before):
    def f(a):
        a = [x for x in a if x != flag]
        k = a.index(before)
        return a[:k] + [flag] + a[k:]
    return f


MA = [
    # name, arm, seed, kwargs, any-of substrings the refusal must contain
    ("ego probe reads off on A (the apply was deleted)", "dagoaccorc", 0, dict(mutate=sc_set(["ego_perturb_probe"], {"ego_perturb": "off", "pass": True})), ["the ego probe ran"]),
    ("ego probe pass False", "dagoaccorc", 0, dict(mutate=sc_set(["ego_perturb_probe", "pass"], False)), ["ego_perturb_probe"]),
    ("ego probe: the perturbed batch moves no gradient", "dagoaccorc", 0, dict(mutate=chain_mut(sc_set(["ego_perturb_probe", "grad_diff_all"], 0.0), sc_set(["ego_perturb_probe", "loss_diff"], 0.0))), ["does not reach the loss"]),
    ("ego probe: control_p0_grad_diff != 0", "dagoaccorc", 0, dict(mutate=sc_set(["ego_perturb_probe", "control_p0_grad_diff"], 1e-7)), ["control_p0_grad_diff"]),
    ("ego probe: control_p0_loss_diff != 0", "dagoaccorc", 0, dict(mutate=sc_set(["ego_perturb_probe", "control_p0_loss_diff"], 1e-9)), ["control_p0_loss_diff"]),
    ("ego probe: control_prior_grad_diff != 0", "dagoaccorc", 0, dict(mutate=sc_set(["ego_perturb_probe", "control_prior_grad_diff"], 1e-9)), ["control_prior_grad_diff"]),
    ("ego probe: control_repeat_grad_diff != 0", "dagoaccorc", 0, dict(mutate=sc_set(["ego_perturb_probe", "control_repeat_grad_diff"], 1e-9)), ["control_repeat_grad_diff"]),
    ("ego probe: p = 0 is not the identity object", "dagoaccorc", 0, dict(mutate=sc_set(["ego_perturb_probe", "control_p0_same_object"], False)), ["not the identity"]),
    ("ego probe: the WM auxiliary's gradient moved", "dagoaccorc", 0, dict(mutate=sc_set(["ego_perturb_probe", "control_wm_grad_diff"], 0.5)), ["world-model auxiliary"]),
    ("ego probe: G-LIVE on the perturbed batch failed", "dagoaccorc", 0, dict(mutate=sc_set(["ego_perturb_probe", "live_perturbed", "pass"], False)), ["G-LIVE on the PERTURBED"]),
    ("ego probe: the frozen prior moved on the perturbed batch", "dagoaccorc", 0, dict(mutate=sc_set(["ego_perturb_probe", "live_perturbed", "control_prior_grad_sum"], 1e-9)), ["G-LIVE on the PERTURBED"]),
    ("ego probe: no eligible window", "dagoaccorc", 0, dict(mutate=sc_set(["ego_perturb_probe", "n_selected_at_p1"], 0)), ["no eligible window"]),
    ("ego probe: declared p 0.1", "dagoaccorc", 0, dict(mutate=sc_set(["ego_perturb_probe", "p_declared"], 0.1)), ["the ego probe ran"]),
    ("composed probe batch has an overlap", "dagoaccorc", 0, dict(mutate=sc_set(["ego_perturb_probe", "ego_dagger_overlap_n"], 3)), ["ego_dagger_overlap_n"]),
    ("built ego p 0.1", "dagoaccorc", 0, dict(mutate=sc_set(["levers", "ego_perturb", "ego_perturb_p"], 0.1)), ["built ego_perturb ego_perturb_p"]),
    ("built vscale changed", "dagoaccorc", 0, dict(mutate=sc_set(["levers", "ego_perturb", "ego_perturb_vscale"], [0.5, 0.9])), ["built ego_perturb ego_perturb_vscale"]),
    ("built ego seed != S", "dagoaccorc", 1, dict(mutate=sc_set(["levers", "ego_perturb", "ego_perturb_seed"], 7)), ["built ego_perturb ego_perturb_seed"]),
    ("built yaw-rate std 0", "dagoaccorc", 0, dict(mutate=sc_set(["levers", "ego_perturb", "ego_perturb_yawrate_std"], 0.0)), ["ego_perturb_yawrate_std"]),
    ("composition flag not built on A (False)", "dagoaccorc", 0, dict(mutate=sc_set(["levers", "ego_perturb_with_dagger"], False)), ["ego_perturb_with_dagger"]),
    ("composition flag absent from the built levers on A", "dagoaccorc", 0, dict(mutate=sc_del(["levers", "ego_perturb_with_dagger"])), ["ego_perturb_with_dagger"]),
    ("REP builds the composition flag", "dagoaccorcrep", 0, dict(mutate=sc_set(["levers", "ego_perturb_with_dagger"], True)), ["ego_perturb_with_dagger"]),
    ("REP built ego p 0.3", "dagoaccorcrep", 0, dict(mutate=sc_set(["levers", "ego_perturb", "ego_perturb_p"], 0.3)), ["built ego_perturb ego_perturb_p"]),
    ("REP: the ego probe is ON", "dagoaccorcrep", 1, dict(mutate=lambda sc: sc.update(ego_perturb_probe=W.load(W.EGO_FIX)["ego_perturb_probe"])), ["should read 'off'"]),
    ("swap probe fails", "dagoaccorc", 1, dict(mutate=sc_set(["dagger_probe", "pass"], False)), ["dagger_probe"]),
    ("swap probe: no gradient", "dagoaccorcrep", 0, dict(mutate=chain_mut(sc_set(["dagger_probe", "grad_diff_all"], 0.0), sc_set(["dagger_probe", "loss_diff"], 0.0))), ["swap does not reach the loss"]),
    ("swap probe: control_p0_loss_diff != 0", "dagoaccorc", 0, dict(mutate=sc_set(["dagger_probe", "control_p0_loss_diff"], 1e-6)), ["dagger probe control"]),
    ("swap probe: p = 0 not the identity", "dagoaccorc", 0, dict(mutate=sc_set(["dagger_probe", "control_p0_same_object"], False)), ["dagger probe: p = 0"]),
    ("swap probe: G-LIVE on the swapped batch failed", "dagoaccorcrep", 1, dict(mutate=sc_set(["dagger_probe", "live_swapped", "pass"], False)), ["SWAPPED batch"]),
    ("swap probe: frame-change error 1e-3", "dagoaccorc", 0, dict(mutate=sc_set(["dagger_probe", "rc_max_abs_err"], 1e-3)), ["frame-change error"]),
    ("built dagger_p 0.2", "dagoaccorc", 0, dict(mutate=sc_set(["levers", "dagger_p"], 0.2)), ["built dagger_p"]),
    ("built bank sha256 differs", "dagoaccorcrep", 0, dict(mutate=sc_set(["levers", "dagger_bank_sha256"], "0" * 64)), ["bank sha256"]),
    ("K-B2: the bank was made by another checkpoint", "dagoaccorc", 0, dict(mutate=sc_set(["info", "dagger", "kb2", "ckpt_sha256"], "1" * 64)), ["K-B2"]),
    ("COVERAGE blocked", "dagoaccorc", 0, dict(mutate=sc_set(["info", "dagger", "coverage", "blocked"], True)), ["COVERAGE"]),
    ("K-B3: overlap with eval-141", "dagoaccorc", 0, dict(mutate=sc_set(["info", "dagger", "kb3", "overlap_with_eval"], 5)), ["K-B3"]),
    ("K-B3: another TRAIN window list", "dagoaccorcrep", 1, dict(mutate=sc_set(["info", "dagger", "kb3", "window_list_sha256"], "2" * 64)), ["K-B3"]),
    ("acc_probe missing", "dagoaccorc", 0, dict(mutate=sc_del(["acc_probe"])), ["acc_probe"]),
    ("acc_probe: read_o_grad_diff 0 on a TRAINED head", "dagoaccorc", 1, dict(mutate=sc_set(["acc_probe", "read_o_grad_diff"], 0.0)), ["read_o_grad_diff"]),
    ("acc_probe: control_prior_grad_diff != 0", "dagoaccorcrep", 0, dict(mutate=sc_set(["acc_probe", "control_prior_grad_diff"], 1e-6)), ["acc_probe control"]),
    ("built w_acc 40", "dagoaccorc", 0, dict(mutate=sc_set(["levers", "w_acc"], 40.0)), ["w_acc"]),
    ("declared_vs_built FAIL", "dagoaccorc", 0, dict(mutate=sc_set(["declared_vs_built"], "FAIL")), ["declared_vs_built"]),
    ("G-LIVE: the prior moved", "dagoaccorcrep", 1, dict(mutate=sc_set(["live", "control_prior_grad_sum"], 1e-6)), ["live "]),
    ("G-LIVE: a dead parameter group", "dagoaccorc", 1, dict(mutate=sc_set(["live", "dead_params"], ["adapter"])), ["dead set"]),
    ("K-INIT: init checkpoint is another file", "dagoaccorc", 0, dict(mutate=chain_mut(sc_set(["levers", "init_ckpt", "sha256"], "3" * 64), sc_set(["info", "init_ckpt", "sha256"], "3" * 64))), ["does not start from M2b"]),
    ("K-INIT: init not applied", "dagoaccorcrep", 0, dict(mutate=chain_mut(sc_set(["levers", "init_ckpt", "applied"], False), sc_set(["info", "init_ckpt", "applied"], False))), ["not applied"]),
    ("K-INIT: init path is the other seed's", "dagoaccorc", 0, dict(mutate=chain_mut(sc_set(["levers", "init_ckpt", "path"], "D:\\x\\M2b_s1\\ckpt_last.pt"), sc_set(["info", "init_ckpt", "path"], "D:\\x\\M2b_s1\\ckpt_last.pt"))), ["init_ckpt path"]),
    ("steps 2000", "dagoaccorc", 0, dict(mutate=sc_set(["steps_total"], 2000)), ["steps"]),
    ("lr schedule 3e-4", "dagoaccorcrep", 1, dict(mutate=sc_set(["levers", "lr_schedule", "lr"], 3e-4)), ["lr schedule"]),
    ("p_gt 1.0 -> 0.2", "dagoaccorc", 0, dict(mutate=sc_set(["levers", "p_gt_start"], 1.0)), ["p_gt"]),
    ("n_clips 4000", "dagoaccorc", 0, dict(mutate=sc_set(["info", "n_clips"], 4000)), ["clips / holdout"]),
    ("holdout is another set", "dagoaccorcrep", 0, dict(mutate=sc_set(["info", "excluded_sha256"], "4" * 64)), ["RA-1 holdout"]),
    ("TRAIN list differs from DAG-OACC's", "dagoaccorc", 0, dict(mutate=sc_set(["info", "train_sha256"], "5" * 64)), ["TRAIN list"]),
    ("w_cons moved (a second lever)", "dagoaccorc", 0, dict(mutate=sc_set(["levers", "w", "w_cons"], 2.0)), ["w_tac / w_cons"]),
    ("parameter groups differ from DAG-OACC's", "dagoaccorc", 0, dict(mutate=lambda sc: sc["levers"]["params"].update({sorted(sc["levers"]["params"])[0]: sc["levers"]["params"][sorted(sc["levers"]["params"])[0]] + 1})), ["parameter group"]),
    ("window set differs", "dagoaccorc", 0, dict(mutate=sc_set(["info", "n_windows_all_clips"], 1)), ["n_windows_all_clips"]),
    ("lru not recorded", "dagoaccorc", 0, dict(mutate=sc_del(["info", "lru"])), ["info.lru"]),
    ("o_interface cut", "dagoaccorc", 0, dict(mutate=sc_set(["levers", "o_interface"], "cut")), ["o_interface"]),
    ("code_mix moved", "dagoaccorc", 0, dict(mutate=sc_set(["levers", "code_mix"], "mix")), ["code_mix"]),
    # ---- the planned lines
    ("A line lacks --ego-perturb-with-dagger", "dagoaccorc", 0, dict(line_mutate=argv_del(G.COMPOSE, 0)), [COMPOSE_MSG := "does not name --ego-perturb-with-dagger exactly once"]),
    ("A line names the flag twice", "dagoaccorc", 0, dict(line_mutate=argv_add([G.COMPOSE])), ["exactly once"]),
    ("A line has the flag BEFORE --w-acc (not the last token)", "dagoaccorc", 1, dict(line_mutate=move_flag_before(G.COMPOSE, "--w-acc")), ["ONE-LEVER"]),
    ("A line: vscale 0.5 0.9", "dagoaccorc", 0, dict(line_mutate=argv_edit(["--ego-perturb-vscale", "0.5", "1.0"], ["--ego-perturb-vscale", "0.5", "0.9"])), ["--ego-perturb-vscale"]),
    ("A line: ego seed != S", "dagoaccorc", 1, dict(line_mutate=argv_edit(["--ego-perturb-seed", "1"], ["--ego-perturb-seed", "0"])), ["--ego-perturb-seed"]),
    ("A line: p 0.1 (the dose fallback B, not queued)", "dagoaccorc", 0, dict(line_mutate=argv_edit(["--ego-perturb-p", "0.3"], ["--ego-perturb-p", "0.1"])), ["--ego-perturb-p"]),
    ("A line: --dagger-p 0.2", "dagoaccorc", 0, dict(line_mutate=argv_edit(["--dagger-p", "0.1"], ["--dagger-p", "0.2"])), ["--dagger-p"]),
    ("A line: --dagger-p 0 (the swap is off)", "dagoaccorc", 1, dict(line_mutate=argv_edit(["--dagger-p", "0.1"], ["--dagger-p", "0"])), ["--dagger-p"]),
    ("A line: another bank sha256", "dagoaccorc", 0, dict(line_mutate=lambda a: [("0" * 64 if x == W.load(W.OACC / "M2b_s0_dagoacc" / "argv.json")["argv"][W.load(W.OACC / "M2b_s0_dagoacc" / "argv.json")["argv"].index("--dagger-bank-sha256") + 1] else x) for x in a]), ["--dagger-bank-sha256"]),
    ("A line: --w-acc 50", "dagoaccorc", 0, dict(line_mutate=argv_edit(["--w-acc", "49.6969"], ["--w-acc", "50"])), ["--w-acc"]),
    ("A line: --dagger-seed other", "dagoaccorc", 1, dict(line_mutate=argv_edit(["--dagger-seed", "1"], ["--dagger-seed", "0"])), ["--dagger-seed"]),
    ("A line: a second lever (--hist 10)", "dagoaccorc", 0, dict(line_mutate=lambda a: a[:a.index("--w-acc")] + ["--hist", "10"] + a[a.index("--w-acc"):]), ["a second lever"]),
    ("A line: steps 6000", "dagoaccorc", 0, dict(line_mutate=argv_edit(["--steps", "3000"], ["--steps", "6000"])), ["--steps"]),
    ("A line: init is the other seed's M2b", "dagoaccorc", 0, dict(line_mutate=lambda a: [x.replace("M2b_s0", "M2b_s1") if "ckpt_last" in x else x for x in a]), ["--init-ckpt"]),
    ("REP line: --seed 0 (not 10)", "dagoaccorcrep", 0, dict(line_mutate=argv_edit(["--seed", "10"], ["--seed", "0"])), ["--seed"]),
    ("REP line: --seed 11 for s0", "dagoaccorcrep", 0, dict(line_mutate=argv_edit(["--seed", "10"], ["--seed", "11"])), ["--seed"]),
    ("REP line names the composition flag", "dagoaccorcrep", 0, dict(line_mutate=argv_add([G.COMPOSE])), ["must be absent"]),
    ("REP line names a law flag at p 0", "dagoaccorcrep", 1, dict(line_mutate=lambda a: a[:a.index("--w-acc")] + ["--ego-perturb-vmax", "8.0"] + a[a.index("--w-acc"):]), ["names --ego-perturb-vmax on REP"]),
    ("REP line: --ego-perturb-p 0.3", "dagoaccorcrep", 0, dict(line_mutate=argv_edit(["--ego-perturb-p", "0"], ["--ego-perturb-p", "0.3"])), ["--ego-perturb-p"]),
    # ---- the gate / pin
    ("the pin is not the planned training line", "dagoaccorc", 0, dict(pin_mutate=argv_edit(["--lr", "1e-4"], ["--lr", "2e-4"])), ["THE PIN IS NOT THE PLANNED"]),
    ("the gate line lacks --selfcheck-only", "dagoaccorcrep", 0, dict(gate_mutate=lambda a: [x for x in a if x != "--selfcheck-only"]), ["--selfcheck-only exactly once"]),
    ("the gate argv differs from the training argv (--batch 32)", "dagoaccorc", 0, dict(gate_mutate=argv_add(["--batch", "32"], at=-1)), ["gate argv is not the training argv"]),
    ("the selfcheck's argv_sha is another line's", "dagoaccorc", 1, dict(gate_sha_override="0" * 16), ["gate ran an argv other than"]),
]
for i, (name, arm, s, kw, subs) in enumerate(MA):
    d = W.build(fresh(f"ma{i}"), arm, s, **kw)
    try:
        bad = quiet(W.run_check, d)
    except Exception as e:                  # a crash is a refusal of a kind, but not the registered one: count it as NOT killed by the intended check
        bad = [f"CRASH {type(e).__name__}: {e}"]
    hit = [b for b in bad if any(x in b for x in subs)]
    M(f"MA{i + 1:02d} {arm} s{s}: {name}", bool(bad) and bool(hit), bad if not hit else "")

# ====================================================================== K. run-record, reached, one-lever (chain checks)
print("== K: run record, K-REACH, one-lever")


def run_dir(root, arm, seed, mut_summary=None, mut_argv=None, mut_sc=None, train_mut=None):
    root.mkdir(parents=True, exist_ok=True)
    base = W.OACC / f"M2b_s{seed}_dagoacc"
    train = W.arm_argv(arm, seed)
    if train_mut:
        train = train_mut(list(train))
    sha16 = hashlib.sha256(" ".join(train).encode()).hexdigest()[:16]
    sm = json.load(open(base / "summary.json", encoding="utf-8"))
    sc = json.load(open(base / "selfcheck.json", encoding="utf-8"))
    p = G.ARMS[arm]
    tseed = 10 + seed if arm == "dagoaccorcrep" else seed
    sm.update(argv_sha=sha16, seed=tseed)
    sc.update(argv_sha=sha16)
    if p > 0.0:
        sm["levers"]["ego_perturb"] = {**sm["levers"]["ego_perturb"], "ego_perturb_p": p, **G.LAW_VALUES, "ego_perturb_seed": seed}
        sm["levers"]["ego_perturb_with_dagger"] = True
        sm.update(ego_perturb_n=24000, ego_perturb_eligible=80000, ego_dagger_overlap_n=0)
    if mut_summary:
        mut_summary(sm)
    if mut_sc:
        mut_sc(sc)
    aj = {"argv": train, "sha": sha16}
    if mut_argv:
        aj = mut_argv(aj)
    for n, o in (("summary.json", sm), ("selfcheck.json", sc), ("argv.json", aj)):
        (root / n).write_text(json.dumps(o), encoding="utf-8", newline="\n")
    for n in ("train_line.txt", "pin_line.txt"):
        (root / n).write_text(W.line(train), encoding="utf-8", newline="\n")
    return root


class NS:
    def __init__(self, **k):
        self.__dict__.update(k)


def rr(root, arm, seed, **kw):
    d = W.build(fresh("rr_w"), arm, seed)
    return quiet(C.run_record, NS(run_dir=str(root), train_line=str(root / "train_line.txt"), pin_line=str(root / "pin_line.txt"), arm=arm, seed=str(seed), banked_init_sha=d["banked_init_sha"],
                                  bank_sha=d["bank_sha"], w_acc=W.W_ACC, steps="3000", expect="", **kw))


for arm in ("dagoaccorc", "dagoaccorcrep"):
    for s in (0, 1):
        T(f"K1 {arm} s{s}: a run record built from DAG-OACC's REAL summary / selfcheck passes", rr(run_dir(fresh(f"k_{arm}_{s}"), arm, s), arm, s) == [])
MK = [
    ("argv.json is not the planned argv", "dagoaccorc", 0, dict(mut_argv=lambda a: {**a, "argv": a["argv"][:-1]}), ["argv.json is not the planned"]),
    ("summary argv_sha differs", "dagoaccorcrep", 1, dict(mut_summary=lambda sm: sm.update(argv_sha="0" * 16)), ["summary.json argv_sha"]),
    ("run not done", "dagoaccorc", 0, dict(mut_summary=lambda sm: sm.update(done=False)), ["done is not true"]),
    ("steps 2999", "dagoaccorc", 1, dict(mut_summary=lambda sm: sm.update(steps=2999)), ["steps"]),
    ("REP trained with seed 0 (not 10)", "dagoaccorcrep", 0, dict(mut_summary=lambda sm: sm.update(seed=0)), ["arm / seed"]),
    ("run's own selfcheck not a PASS", "dagoaccorc", 0, dict(mut_sc=lambda sc: sc.update(declared_vs_built="FAIL")), ["not a PASS"]),
    ("K-INIT: init checkpoint is another file", "dagoaccorc", 0, dict(mut_summary=lambda sm: sm["levers"]["init_ckpt"].update(sha256="6" * 64)), ["K-INIT"]),
    ("init not applied", "dagoaccorcrep", 1, dict(mut_summary=lambda sm: sm["levers"]["init_ckpt"].update(applied=False)), ["not applied"]),
    ("built ego p 0.1 on A", "dagoaccorc", 0, dict(mut_summary=lambda sm: sm["levers"]["ego_perturb"].update(ego_perturb_p=0.1)), ["built ego_perturb ego_perturb_p"]),
    ("built ego seed wrong", "dagoaccorc", 1, dict(mut_summary=lambda sm: sm["levers"]["ego_perturb"].update(ego_perturb_seed=0)), ["ego_perturb_seed"]),
    ("composition flag not built on A", "dagoaccorc", 0, dict(mut_summary=lambda sm: sm["levers"].update(ego_perturb_with_dagger=False)), ["on A (must be True)"]),
    ("composition flag built on REP", "dagoaccorcrep", 0, dict(mut_summary=lambda sm: sm["levers"].update(ego_perturb_with_dagger=True)), ["on REP"]),
    ("built dagger_p 0", "dagoaccorc", 0, dict(mut_summary=lambda sm: sm["levers"].update(dagger_p=0.0)), ["built dagger_p"]),
    ("another bank built", "dagoaccorcrep", 0, dict(mut_summary=lambda sm: sm["levers"].update(dagger_bank_sha256="7" * 64)), ["dagger_bank_sha256"]),
    ("w_acc 0 built", "dagoaccorc", 1, dict(mut_summary=lambda sm: sm["levers"].update(w_acc=0.0)), ["w_acc"]),
]
for i, (name, arm, s, kw, subs) in enumerate(MK):
    bad = rr(run_dir(fresh(f"mk{i}"), arm, s, **kw), arm, s)
    M(f"MK{i + 1:02d} run-record {arm} s{s}: {name}", bool(bad) and any(x in b for b in bad for x in subs), bad)


def reach(arm, mut):
    r = fresh("reach")
    sm = json.load(open(W.OACC / "M2b_s0_dagoacc" / "summary.json", encoding="utf-8"))
    if G.ARMS[arm] > 0.0:
        sm.update(ego_perturb_n=24000, ego_perturb_eligible=80000, ego_dagger_overlap_n=0)
        sm.setdefault("levers", {})["ego_perturb_with_dagger"] = True
    if mut:
        mut(sm)
    (r / "summary.json").write_text(json.dumps(sm), encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        return C.reached(NS(summary=str(r / "summary.json"), arm=arm))


T("K2 K-REACH REP passes on DAG-OACC's REAL banked summary (swap 14688/145037 = 0.101, ego_perturb_n exactly 0, flag absent)", reach("dagoaccorcrep", None) == 0)
T("K3 K-REACH A passes: swap 0.1, ego 0.3 of the non-swapped eligible, overlap exactly 0, flag True", reach("dagoaccorc", None) == 0)
MR = [
    ("A: swap fraction 0.2", "dagoaccorc", lambda sm: sm.update(dagger_n=29000, dagger_eligible=145037)),
    ("A: swap fraction 0 (the swap never ran)", "dagoaccorc", lambda sm: sm.update(dagger_n=0)),
    ("A: ego fraction 0.5", "dagoaccorc", lambda sm: sm.update(ego_perturb_n=40000)),
    ("A: ego_perturb_n 0 (the perturbation never ran)", "dagoaccorc", lambda sm: sm.update(ego_perturb_n=0)),
    ("A: ego fraction 0.33 (outside 0.3 +- 0.02)", "dagoaccorc", lambda sm: sm.update(ego_perturb_n=26400)),
    ("A: overlap 1 (a sample both swapped and perturbed)", "dagoaccorc", lambda sm: sm.update(ego_dagger_overlap_n=1)),
    ("A: overlap absent", "dagoaccorc", lambda sm: sm.pop("ego_dagger_overlap_n")),
    ("A: composition flag False", "dagoaccorc", lambda sm: sm["levers"].update(ego_perturb_with_dagger=False)),
    ("A: composition flag absent", "dagoaccorc", lambda sm: sm["levers"].pop("ego_perturb_with_dagger")),
    ("A: eligible 0", "dagoaccorc", lambda sm: sm.update(ego_perturb_eligible=0)),
    ("REP: ego_perturb_n 5 (a perturbation leaked into the replicate)", "dagoaccorcrep", lambda sm: sm.update(ego_perturb_n=5)),
    ("REP: swap fraction 0.3", "dagoaccorcrep", lambda sm: sm.update(dagger_n=43500)),
    ("REP: swap never ran", "dagoaccorcrep", lambda sm: sm.update(dagger_n=0)),
    ("REP: composition flag True", "dagoaccorcrep", lambda sm: sm.setdefault("levers", {}).update(ego_perturb_with_dagger=True)),
    ("REP: ego eligible 0", "dagoaccorcrep", lambda sm: sm.update(ego_perturb_eligible=0)),
]
for i, (name, arm, mut) in enumerate(MR):
    M(f"MR{i + 1:02d} K-REACH {arm}: {name}", reach(arm, mut) == 1)

# one-lever
w = W.build(fresh("ol"), "dagoaccorc", 0)
for arm in ("dagoaccorc", "dagoaccorcrep"):
    for s in (0, 1):
        w = W.build(fresh(f"ol{arm}{s}"), arm, s)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            rc = C.one_lever(NS(ref_argv_json=w["ref_argv_json"], line=w["train_line"], arm=arm, seed=str(s)))
        T(f"K4 one-lever {arm} s{s}: ONE-LEVER on the registered line", rc == 0 and buf.getvalue().strip() == "ONE-LEVER")
for i, (name, arm, s, kw, subs) in enumerate([x for x in MA if "line_mutate" in x[3]]):
    w = W.build(fresh(f"olm{i}"), arm, s, **kw)
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        rc = C.one_lever(NS(ref_argv_json=w["ref_argv_json"], line=w["train_line"], arm=arm, seed=str(s)))
    M(f"MK-OL{i + 1:02d} one-lever {arm} s{s}: {name}", rc == 1 and buf.getvalue().strip() == "NOT-ONE-LEVER")

# ====================================================================== Q. the queue
print("== Q: the queue patch")


def cmdrun(queue, args, env=None, dryrun=True, sha="auto", wacc=W.W_ACC):
    e = {**BOX2, **(env or {})}
    if sha == "auto":      # the seed's own iteration-1 bank sha256 (DAG-OACC's argv.json), by the seed in the stage arguments
        tk = args.split()
        s_ = tk[2] if len(tk) > 2 and tk[2] in ("0", "1") else "0"
        a_ = W.load(W.OACC / f"M2b_s{s_}_dagoacc" / "argv.json")["argv"]
        sha = a_[a_.index("--dagger-bank-sha256") + 1]
    lines = ["@echo off"]
    if dryrun:
        lines.append("set V8F_DRYRUN=1")
        if sha:
            lines.append(f'set "DAGSHA_DRY={sha}"')
        if wacc:
            lines.append(f'set "WACCDAG_DRY={wacc}"')
    lines += [f'set "{k}={v}"' for k, v in e.items()]
    lines += [f'call "{queue}" {args}', "exit /b %errorlevel%"]
    SCR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".cmd", delete=False, dir=str(SCR), newline="") as fh:
        fh.write("\r\n".join(lines) + "\r\n")
        wr = fh.name
    try:
        p = subprocess.run(["cmd.exe", "/c", wr], capture_output=True, stdin=subprocess.DEVNULL, timeout=300)
    finally:
        os.unlink(wr)
    out = (p.stdout or b"").decode("utf-8", "replace").replace("\r", "")
    return p.returncode, [x for x in out.split("\n") if x != ""]


def targv(lines, exe="v8f_mini_train.py"):
    x = [l for l in lines if exe in l]
    if len(x) != 1:
        return None
    t = [y.strip('"') for y in re.findall(r'"[^"]*"|\S+', x[0].strip())]
    return t[next(k for k, y in enumerate(t) if y.lower().endswith(exe)) + 1:]


import build_queue as BQ                                 # noqa: E402
T("Q1 the staged queue == a fresh rebuild from the pinned tip queue (deterministic)", BQ.build() == QUEUE.read_bytes())
dtxt = (HERE / "queue.diff").read_text(encoding="utf-8").split("\n")
T("Q2 queue.diff removes NO line (insertion only)", not [x for x in dtxt if x.startswith("-") and not x.startswith("---")] and sum(1 for x in dtxt if x.startswith("+") and not x.startswith("+++")) == 19)
T("Q3 the staged queue keeps the tip's CRLF on every line", QUEUE.read_bytes().count(b"\n") == QUEUE.read_bytes().count(b"\r\n"))
EXIST = ["gatedag dagoacc 0 0.1", "traindag dagoacc 1 0.1", "evaldag dagoacc 0", "gatedag dag 0 0.1", "traindag dag0 1", "gatedag dag2 0 0.1", "traindag dag20 0", "gatedag flatdag 0 0.1", "traindag flatdag0 1",
         "gatedag dagorc 0", "traindag dagorcrep 1", "traindag dagorc01 0", "evaldag dagorc 1", "waccdag", "gate", "train M2b 0", "eval M2b 0", "gatearm M2b"]
diff_ex = []
for st in EXIST:
    a, b = cmdrun(TIPQ, st), cmdrun(QUEUE, st)
    if a != b:
        diff_ex.append((st, a, b))
T(f"Q4 {len(EXIST)} existing stage invocations print byte-identical output on the tip queue and the patched queue", not diff_ex, diff_ex[:1])
base = {s: W.load(W.OACC / f"M2b_s{s}_dagoacc" / "argv.json")["argv"] for s in (0, 1)}
for s in (0, 1):
    rc, out = cmdrun(QUEUE, f"traindag dagoacc {s} 0.1")
    T(f"Q5 the queue's own DAG-OACC s{s} train line == DAG-OACC's banked argv.json (only --out may differ): the base of the one-lever rule is the real recipe", rc == 0 and G.drop(targv(out)) == G.drop(base[s]), (rc, out[:2]))
for arm in ("dagoaccorc", "dagoaccorcrep"):
    for s in (0, 1):
        exp = G.expected_from_ref(base[s], arm, s)
        for st in ("gatedag", "traindag"):
            rc, out = cmdrun(QUEUE, f"{st} {arm} {s}")
            a = targv(out)
            if st == "gatedag" and a:
                a = [x for x in a if x != "--selfcheck-only"]
            T(f"Q6 {st} {arm} {s}: the printed line == DAG-OACC's banked argv + the registered change (token for token)", rc == 0 and a is not None and G.drop(a) == exp, (rc, (a or [])[-8:]))
        rc, out = cmdrun(QUEUE, f"evaldag {arm} {s}")
        e = targv(out, "v8f_mini_eval.py")
        T(f"Q7 evaldag {arm} {s}: the standard eval of its own run / tag", rc == 0 and e is not None and f"M2b-{arm}_s{s}" in e and f"M2b_s{s}_{arm}" in " ".join(e), out[:1])
for st in ("gatedag dagoaccorc 0 0.1", "traindag dagoaccorcrep 1 0.3", "gatedag dagoaccorc 2", "gatedag dagoaccorc"):
    rc, out = cmdrun(QUEUE, st)
    T(f"Q8 REFUSED (rc 2, no trainer line): {st}", rc == 2 and targv(out) is None and any("REFUSED" in x for x in out), (rc, out[:2]))

print("== Q-mut: deliberate regressions of the queue stage (each must break the registered line)")
qtxt = QUEUE.read_bytes().decode("utf-8")


def qmut(name, old, new, arms_expect_bad):
    assert qtxt.count(old) == 1, (name, qtxt.count(old))
    p = SCR / "qmut.cmd"
    p.write_bytes(qtxt.replace(old, new).encode("utf-8"))
    bad = []
    for arm in ("dagoaccorc", "dagoaccorcrep"):
        for s in (0, 1):
            exp = G.expected_from_ref(base[s], arm, s)
            for st in ("gatedag", "traindag"):
                rc, out = cmdrun(p, f"{st} {arm} {s}")
                a = targv(out)
                if st == "gatedag" and a:
                    a = [x for x in a if x != "--selfcheck-only"]
                if rc != 0 or a is None or G.drop(a) != exp:
                    bad.append((arm, s, st))
    arms = {x[0] for x in bad}
    M(f"MQ {name}: breaks {sorted(arms)}", bool(bad) and arms >= set(arms_expect_bad), bad[:2])


qmut("the composition flag is not set on A", 'set "DAGORCX= --ego-perturb-with-dagger")', 'set "DAGORCX=")', ["dagoaccorc"])
qmut("the REP's trainer seed is not 1<S>", 'if /i "%DAGS%"=="dagoaccorcrep" set "DAGSEED=1%~2"', 'rem', ["dagoaccorcrep"])
qmut("A's ego seed is a constant 0", '--ego-perturb-seed %~2" & set "DAGORCX=', '--ego-perturb-seed 0" & set "DAGORCX=', ["dagoaccorc"])
qmut("A's ego p is 0.1", 'if /i "%DAGS%"=="dagoaccorc" (set "DAGORCP=0.3"', 'if /i "%DAGS%"=="dagoaccorc" (set "DAGORCP=0.1"', ["dagoaccorc"])
qmut("A is missing its law flags", 'if /i "%DAGS%"=="dagoaccorc" (set "DAGORCP=0.3" & set "DAGORCL= --ego-perturb-vscale 0.5 1.0', 'if /i "%DAGS%"=="dagoaccorc" (set "DAGORCP=0.3" & set "DAGORCL=', ["dagoaccorc"])
qmut("w_acc is a constant 50", '--dagger-seed %~2 --w-acc %WACCDAG%%DAGORCX%"', '--dagger-seed %~2 --w-acc 50%DAGORCX%"', ["dagoaccorc", "dagoaccorcrep"])
qmut("the swap p is 0 (DAGP not taken from :dagoaccarm)", '--ego-perturb-p %DAGORCP%%DAGORCL% --dagger-p %DAGP%', '--ego-perturb-p %DAGORCP%%DAGORCL% --dagger-p 0', ["dagoaccorc", "dagoaccorcrep"])
qmut("the init is DAG's checkpoint, not M2b's", 'set "DAGF=--init-ckpt "%RUNSM%\\M2b_s%~2\\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p %DAGORCP%', 'set "DAGF=--init-ckpt "%RUNSM%\\M2b_s%~2_dag\\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p %DAGORCP%', ["dagoaccorc", "dagoaccorcrep"])
pold = 'if not "%~3"=="" (echo REFUSED: %DAGS% takes no P, the dose is fixed by PREREG_V8F_DAG_ORC Amendment 1, got "%~3" & exit /b 2)'
assert qtxt.count(pold) == 1
p = SCR / "qmut_p.cmd"
p.write_bytes(qtxt.replace(pold, "rem").encode("utf-8"))
rc, out = cmdrun(p, "gatedag dagoaccorc 0 0.2")
M("MQ the P refusal deleted: 'gatedag dagoaccorc 0 0.2' prints a trainer line again -> the Q8 refusal test would go RED", targv(out) is not None and rc == 0)
qmut("the :dagarm hand-off is deleted for dagoaccorcrep", 'if /i "%DAGS%"=="dagoaccorcrep" goto :dagoaccorcarm', 'rem', ["dagoaccorcrep"])
qmut("the :dagname acceptance of dagoaccorc is deleted", 'if /i "%~1"=="dagoaccorc" set "DAGS=dagoaccorc"', 'rem', ["dagoaccorc"])

# ====================================================================== R. the queue run NOT dry on a fake box (stub trainer)
print("== R: the queue's real (non-dry) gate / train path on a fake box")


def fake_box(root):
    shutil.rmtree(root, ignore_errors=True)
    repo, runsm, cache, v9 = root / "repo", root / "runs", root / "cache_full", root / "v9"
    (repo / "stack" / "scripts").mkdir(parents=True)
    hj = repo / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-refav1-hj-mini" / "raw" / "p10_yaw4512"
    hj.mkdir(parents=True)
    (hj / "p10_prior.pt").write_bytes(b"prior")
    (hj / "p10_eval_pred6.npy").write_bytes(b"pred")
    pk = repo / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini" / "raw"
    pk.mkdir(parents=True)
    (pk / "eval_grid_windows.json").write_text("{}")
    for dd in ("train", "eval"):
        (cache / dd).mkdir(parents=True)
        (cache / dd / "manifest.jsonl").write_text("{}\n")
    (cache / "native_tracks.pt").write_bytes(b"t")
    runsm.mkdir(parents=True)
    for s in (0, 1):
        (runsm / "dagger_bank" / f"s{s}").mkdir(parents=True)
        (runsm / "dagger_bank" / f"s{s}" / "bank.json").write_text("{}")
    (runsm / "dagoacc").mkdir()
    shutil.copy(W.OACC / "wacc" / "w_acc.txt", runsm / "dagoacc" / "w_acc.txt")
    (repo / "stack" / "scripts" / "v8f_mini_train.py").write_text(
        "import json, os, sys\na = sys.argv[1:]\nout = a[a.index('--out') + 1]\nos.makedirs(out, exist_ok=True)\nopen(os.path.join(out, 'stub_argv.json'), 'w').write(json.dumps(a))\n"
        "if '--selfcheck-only' not in a:\n    open(os.path.join(out, 'summary.json'), 'w').write('{}')\n    open(os.path.join(out, 'ckpt_last.pt'), 'w').write('x')\n")
    (repo / "stack" / "scripts" / "v8f_dagger_queue_tool.py").write_text("import sys\nprint('0' * 64)\n")
    return {"REPO": str(repo), "PY": sys.executable, "CACHEFULL": str(cache), "V9": str(v9), "RUNSM": str(runsm)}


env = fake_box(SCR / "fakebox")
runs = Path(env["RUNSM"])
rc, out = cmdrun(QUEUE, "gatedag dagoaccorc 0", env=env, dryrun=False, sha=None, wacc=None)
T("R1 gatedag dagoaccorc 0 (real run): rc 0 and the gate ran", rc == 0, (rc, out[-3:]))
pin = runs / "dagpin" / "M2b_s0_dagoaccorc.train.argv.txt"
ptxt = pin.read_text() if pin.exists() else ""
T("R2 the training argv is PINNED with the perturbation, the composition flag LAST, the swap and w_acc 49.6969, and without --selfcheck-only",
  pin.exists() and "--ego-perturb-p 0.3" in ptxt and ptxt.rstrip().endswith("--ego-perturb-with-dagger") and "--dagger-p 0.1" in ptxt and "--w-acc 49.6969" in ptxt and "--selfcheck-only" not in ptxt, ptxt[-200:])
stub = json.loads((runs / "gate_M2b-dagoaccorc_s0" / "stub_argv.json").read_text()) if (runs / "gate_M2b-dagoaccorc_s0" / "stub_argv.json").exists() else []
T("R3 the gate ran the trainer with --selfcheck-only and the pinned argv (stub saw it)", "--selfcheck-only" in stub and stub[stub.index("--ego-perturb-p") + 1] == "0.3" and stub[stub.index("--dagger-bank-sha256") + 1] == "0" * 64 and "--ego-perturb-with-dagger" in stub and stub.index("--ego-perturb-with-dagger") > stub.index("--w-acc"), stub[-6:])
rc, out = cmdrun(QUEUE, "gatedag dagoaccorc 0", env=env, dryrun=False, sha=None, wacc=None)
T("R4 a second gate with an unchanged argv is accepted (idempotent pin)", rc == 0, out[-2:])
good = pin.read_bytes()
pin.write_bytes(good.replace(b"--ego-perturb-p 0.3", b"--ego-perturb-p 0.1"))
rc, out = cmdrun(QUEUE, "gatedag dagoaccorc 0", env=env, dryrun=False, sha=None, wacc=None)
T("R5 a re-gate against a pin that names another argv is REFUSED", rc == 2 and any("pins another training argv" in x for x in out), (rc, out[-2:]))
rc, out = cmdrun(QUEUE, "traindag dagoaccorc 0", env=env, dryrun=False, sha=None, wacc=None)
T("R6 traindag against a tampered pin is REFUSED", rc == 2 and any("differs from the one the gate tokenised" in x for x in out), (rc, out[-2:]))
pin.write_bytes(good)
rc, out = cmdrun(QUEUE, "traindag dagoaccorcrep 1", env=env, dryrun=False, sha=None, wacc=None)
T("R7 traindag before its gate is REFUSED (no pinned training argv)", rc == 2 and any("no pinned training argv" in x for x in out), (rc, out[-2:]))
rc, out = cmdrun(QUEUE, "gatedag dagoaccorcrep 1", env=env, dryrun=False, sha=None, wacc=None)
rc2, out2 = cmdrun(QUEUE, "traindag dagoaccorcrep 1", env=env, dryrun=False, sha=None, wacc=None)
stub = json.loads((runs / "M2b_s1_dagoaccorcrep" / "stub_argv.json").read_text()) if (runs / "M2b_s1_dagoaccorcrep" / "stub_argv.json").exists() else []
T("R8 gate then train REP s1: the trainer ran with --seed 11, --dagger-seed 1, no ego law flag, no composition flag",
  rc == 0 and rc2 == 0 and stub and stub[stub.index("--seed") + 1] == "11" and stub[stub.index("--dagger-seed") + 1] == "1" and "--ego-perturb-with-dagger" not in stub and "--ego-perturb-vmax" not in stub
  and stub[stub.index("--ego-perturb-p") + 1] == "0", (rc, rc2, stub[-8:]))
rc, out = cmdrun(QUEUE, "evaldag dagoaccorc 1", env=env, dryrun=False, sha=None, wacc=None)
T("R9 evaldag before the training finished is REFUSED", rc == 2 and any("has not finished training" in x for x in out), (rc, out[-2:]))
(runs / "dagoacc" / "w_acc.txt").unlink()
rc, out = cmdrun(QUEUE, "gatedag dagoaccorc 1", env=env, dryrun=False, sha=None, wacc=None)
M("MR-R w_acc.txt absent on the box: the gate is REFUSED (no w_acc for DAG-OACC), never run with a guessed weight", rc == 2 and any("no w_acc" in x for x in out), (rc, out[-2:]))
shutil.copy(W.OACC / "wacc" / "w_acc.txt", runs / "dagoacc" / "w_acc.txt")
(runs / "dagger_bank" / "s1" / "bank.json").unlink()
rc, out = cmdrun(QUEUE, "gatedag dagoaccorc 1", env=env, dryrun=False, sha=None, wacc=None)
M("MR-R the iteration-1 bank absent on the box: the gate is REFUSED (no merged bank)", rc == 2 and any("no merged bank" in x for x in out), (rc, out[-2:]))

# ====================================================================== C. the chain script
print("== C: the chain script (check-tree, dry-run, pins, waiter)")


def sh(cmdline, env=None, timeout=300):
    e = {**os.environ, "MSYS_NO_PATHCONV": "0", **(env or {})}
    p = subprocess.run([BASH, "-c", cmdline], capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout, env=e)
    return p.returncode, (p.stdout or b"").decode("utf-8", "replace").replace("\r", "") + (p.stderr or b"").decode("utf-8", "replace").replace("\r", "")


def ship(root):
    shutil.rmtree(root, ignore_errors=True)
    (root / "tmp").mkdir(parents=True)
    for f in ("chain_v8f_dagoaccorc.sh", "gate_dagoaccorc_assert.py", "v8f_dagoaccorc_chain_checks.py", "v8f_guard_dagoaccorc.sh"):
        shutil.copy(HERE / "box2_ship" / f, root / f)
    for f in ("oaf_lib.sh", "v8f_oaf_chain_checks.py"):
        shutil.copy(Path("C:/Users/Admin/box2_ship") / f, root / f)
    return root


def bp(p):
    return str(p).replace("\\", "/")


SD = ship(SCR / "ship")
qf = bp(QUEUE)
rc, out = sh(f'cd "{bp(SD)}" && TMPDIR="{bp(SD)}/tmp" OAF_SHIP_DIR="{bp(SD)}" V8F_QUEUE="{qf}" AFTER_OUT=/c/Users/Admin/box2_ship/logs/chain_v8f_dagoacc_r2.out bash ./chain_v8f_dagoaccorc.sh --dry-run')
T("C1 --dry-run on the patched queue: rc 0, 4 SAME, 4 ONE-LEVER, 4 runs in the registered order", rc == 0 and out.count("-> SAME") + out.count(": SAME") == 4 and out.count(": ONE-LEVER") == 4
  and "== runs (order): dagoaccorc 0 dagoaccorcrep 0 dagoaccorc 1 dagoaccorcrep 1" in out, out[-400:])
# the tree check
G_TIP = "6cd741b63b3b629282d48f3b88d86f99264f9c9f"
MIR = "C:/Users/Admin/tanitad-push"
files = ["stack/scripts/v8f_mini_train.py", "stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tanitad/data/v8f_dagger.py"]


def tree(root, queue_path, trainer_extra="", drop_add_argument=False):
    shutil.rmtree(root, ignore_errors=True)
    for f in files:
        dst = root / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{G_TIP}:{f}"], capture_output=True, check=True).stdout)
    q = root / PK / "code" / "v8f_mini_box2_queue.cmd"
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_bytes(Path(queue_path).read_bytes())
    if trainer_extra:
        with open(root / files[0], "a", encoding="utf-8", newline="\n") as fh:
            fh.write(trainer_extra)
    return root


STUB = ('\n    ap.add_argument("--ego-perturb-with-dagger", dest="ego_perturb_with_dagger", action="store_true")\n'
        'FLAG_CHECKS_STUB = {"ego_perturb_with_dagger": "ego:ego_perturb_with_dagger"}\n_STUB = {"ego_dagger_overlap_n": 0}\n')
rc, out = sh(f'cd "{bp(SD)}" && bash ./chain_v8f_dagoaccorc.sh --check-tree "{bp(tree(SCR / "t_tip", TIPQ))}"')
T("C2 --check-tree on the CURRENT tip (no flag, no queue stage): REFUSED, naming the missing flag", rc == 1 and 'add_argument\\("--ego-perturb-with-dagger"' in out and "lever check: FAILED" in out, out[-300:])
rc, out = sh(f'cd "{bp(SD)}" && bash ./chain_v8f_dagoaccorc.sh --check-tree "{bp(tree(SCR / "t_stub", QUEUE, STUB))}"')
T("C3 --check-tree on a tree patched with the flag stub + the patched queue: PASS (every pattern present)", rc == 0 and "every one of" in out, out[-300:])
rc, out = sh(f'cd "{bp(SD)}" && bash ./chain_v8f_dagoaccorc.sh --check-tree "{bp(tree(SCR / "t_noflag", QUEUE, STUB.replace("ap.add_argument(", "ap.XXX(")))}"')
M("MC01 the tree lacks add_argument(\"--ego-perturb-with-dagger\" but has the literals: REFUSED", rc == 1 and 'add_argument\\("--ego-perturb-with-dagger"' in out, out[-300:])
rc, out = sh(f'cd "{bp(SD)}" && bash ./chain_v8f_dagoaccorc.sh --check-tree "{bp(tree(SCR / "t_noq", TIPQ, STUB))}"')
M("MC02 the tree has the flag but the UNPATCHED queue: REFUSED (no :dagoaccorcarm)", rc == 1 and "dagoaccorcarm" in out, out[-300:])
rc, out = sh(f'cd "{bp(SD)}" && bash ./chain_v8f_dagoaccorc.sh --check-tree "{bp(tree(SCR / "t_noov", QUEUE, STUB.replace("ego_dagger_overlap_n", "x")))}"')
M("MC03 the tree has the flag but no ego_dagger_overlap_n record (K-REACH could not be read): REFUSED", rc == 1 and "ego_dagger_overlap_n" in out, out[-300:])
# a changed helper is a changed chain
for f, why in (("gate_dagoaccorc_assert.py", "reviewed asserter"), ("v8f_dagoaccorc_chain_checks.py", "reviewed checks"), ("oaf_lib.sh", "proven file"), ("v8f_oaf_chain_checks.py", "proven file")):
    sd = ship(SCR / "ship_m")
    with open(sd / f, "ab") as fh:
        fh.write(b"\n# tampered\n")
    rc, out = sh(f'cd "{bp(sd)}" && TMPDIR="{bp(sd)}/tmp" OAF_SHIP_DIR="{bp(sd)}" V8F_QUEUE="{qf}" AFTER_OUT=x bash ./chain_v8f_dagoaccorc.sh --dry-run')
    M(f"MC {f} tampered: the chain STOPs before touching anything ({why})", rc == 1 and "ZZV8FMINI-STOP" in out and why.split()[-1] in out, out[-250:])
# chain-level pin mutants: the dry run must STOP
src = (HERE / "box2_ship" / "chain_v8f_dagoaccorc.sh").read_text(encoding="utf-8")
for name, old, new in (("w_acc pinned as 50", "WACC=49.6969", "WACC=50"), ("seed-0 bank pinned wrong", "BANK1_SHA=(197a68bb", "BANK1_SHA=(297a68bb"), ("runs in another order", 'RUNS=("dagoaccorc 0" "dagoaccorcrep 0"', 'RUNS=("dagoaccorcrep 0" "dagoaccorc 0"')):
    assert src.count(old) == 1, name
    sd = ship(SCR / "ship_m")
    (sd / "chain_v8f_dagoaccorc.sh").write_bytes(src.replace(old, new).encode("utf-8"))
    rc, out = sh(f'cd "{bp(sd)}" && TMPDIR="{bp(sd)}/tmp" OAF_SHIP_DIR="{bp(sd)}" V8F_QUEUE="{qf}" AFTER_OUT=x bash ./chain_v8f_dagoaccorc.sh --dry-run')
    if name == "runs in another order":
        M(f"MC chain mutant: {name}: the dry run prints the wrong order (the registered one is absent)", "== runs (order): dagoaccorc 0 dagoaccorcrep 0 dagoaccorc 1 dagoaccorcrep 1" not in out)
    else:
        M(f"MC chain mutant: {name}: the dry run STOPs (not the one-lever argv)", rc == 1 and "ZZV8FMINI-STOP the planned lines are not the one-lever" in out, out[-250:])
# the real-run preconditions
sd = ship(SCR / "ship_m")
rc, out = sh(f'cd "{bp(sd)}" && TMPDIR="{bp(sd)}/tmp" OAF_SHIP_DIR="{bp(sd)}" bash ./chain_v8f_dagoaccorc.sh')
T("C4 a real run with AFTER_OUT unset STOPs (no default predecessor)", rc == 1 and "AFTER_OUT is not set" in out, out[-250:])
lg = sd / "logs"
lg.mkdir()
(lg / "pred.out").write_text("... \nZZV8FMINI-DONE all\n")
rc, out = sh(f'cd "{bp(sd)}" && TMPDIR="{bp(sd)}/tmp" OAF_SHIP_DIR="{bp(sd)}" AFTER_OUT="{bp(lg)}/pred.out" bash ./chain_v8f_dagoaccorc.sh --wait-only')
T("C5 --wait-only: GO after the predecessor's ZZV8FMINI-DONE with no BOX2_BUSY", rc == 0 and "ZZOAF-WAIT-GO" in out, out[-250:])
(sd / "BOX2_BUSY.txt").write_text("held")
rc, out = sh(f'cd "{bp(sd)}" && TMPDIR="{bp(sd)}/tmp" OAF_SHIP_DIR="{bp(sd)}" OAF_POLL_S=1 OAF_GRACE_S=1 AFTER_OUT="{bp(lg)}/pred.out" timeout 60 bash ./chain_v8f_dagoaccorc.sh --wait-only')
M("MC BOX2_BUSY held after the predecessor's DONE: the wait does NOT GO (HALT)", "ZZOAF-WAIT-GO" not in out and rc != 0, out[-250:])

# ---- the waiter launcher, run on a COPY with scratch paths and a stub chain (the real one writes under box2_ship/logs)
print("== W: the waiter launcher")
wsrc = (HERE / "box2_ship" / "run_dagoaccorc_after_speedread.sh").read_text(encoding="utf-8")
wd = fresh("waiter")
(wd / "ship" / "logs").mkdir(parents=True)
(wd / "pull.out").write_text("pull running\n")
(wd / "ship" / "chain_v8f_dagoaccorc.sh").write_bytes(b'#!/usr/bin/env bash\necho "STUB CHAIN AFTER_OUT=$AFTER_OUT args=$*"\n')
wtxt = wsrc.replace("P=/f/TanitAD-scratch/speedread/pull.out; B=/c/Users/Admin/box2_ship;", f"P={bp(wd)}/pull.out; B={bp(wd)}/ship;").replace("sleep 60", "sleep 1").replace("seq 1 720", "seq 1 6")
wtxt = wtxt.replace("AFTER_OUT=C:/Users/Admin/box2_ship/logs/chain_v8f_dagoacc_r2.out", "AFTER_OUT=C:/Users/Admin/box2_ship/logs/chain_v8f_dagoacc_r2.out")
assert wtxt != wsrc and "/c/Users/Admin/box2_ship" not in wtxt.replace("C:/Users/Admin/box2_ship/logs/chain_v8f_dagoacc_r2.out", "")
(wd / "waiter.sh").write_bytes(wtxt.encode("utf-8"))
rc, out = sh(f'bash "{bp(wd)}/waiter.sh"')
wl = (wd / "ship" / "logs" / "dagoaccorc_waiter.out").read_text()
M("MW the waiter does NOT start the chain while the SPEEDREAD pull has not ended (no marker): it STOPs after its wait", rc == 3 and "ZZDAGOACCORC-WAITER-STOP" in wl and "waiter GO" not in wl, (rc, wl))
(wd / "pull.out").write_text("pull done\nZZSPEEDREAD-PULL-DONE\n")
(wd / "ship" / "BOX2_BUSY.txt").write_text("held")
shutil.rmtree(wd / "ship" / "logs")
(wd / "ship" / "logs").mkdir()
rc, out = sh(f'bash "{bp(wd)}/waiter.sh"')
wl = (wd / "ship" / "logs" / "dagoaccorc_waiter.out").read_text()
M("MW the waiter does NOT start the chain while BOX2_BUSY.txt is present (the pull has ended)", rc == 3 and "waiter GO" not in wl, (rc, wl))
(wd / "ship" / "BOX2_BUSY.txt").unlink()
shutil.rmtree(wd / "ship" / "logs")
(wd / "ship" / "logs").mkdir()
rc, out = sh(f'bash "{bp(wd)}/waiter.sh"')
wl = (wd / "ship" / "logs" / "dagoaccorc_waiter.out").read_text()
co = (wd / "ship" / "logs" / "chain_v8f_dagoaccorc.out").read_text() if (wd / "ship" / "logs" / "chain_v8f_dagoaccorc.out").exists() else ""
T("W1 the waiter starts the chain once the pull has ended AND BOX2_BUSY is absent, with AFTER_OUT = the DAG-OACC r2 log and no SKIP_B variable", rc == 0 and "waiter GO" in wl and "AFTER_OUT=C:/Users/Admin/box2_ship/logs/chain_v8f_dagoacc_r2.out" in co
  and "SKIP_B" not in wtxt, (rc, wl, co))
(wd / "pull.out").write_text("pull stopped\nZZSPEEDREAD-PULL-STOP\n")
shutil.rmtree(wd / "ship" / "logs")
(wd / "ship" / "logs").mkdir()
rc, out = sh(f'bash "{bp(wd)}/waiter.sh"')
T("W2 a ZZSPEEDREAD-PULL-STOP also releases the waiter", rc == 0 and "waiter GO" in (wd / "ship" / "logs" / "dagoaccorc_waiter.out").read_text())

# ====================================================================== summary
npass, nfail = sum(1 for _, ok in RESULTS if ok), sum(1 for _, ok in RESULTS if not ok)
nkill, nsurv = sum(1 for _, ok in MUTANTS if ok), sum(1 for _, ok in MUTANTS if not ok)
print(f"\nSELFTEST: {npass}/{npass + nfail} tests pass; MUTATION: {nkill}/{nkill + nsurv} mutants killed, {nsurv} survivors")
json.dump({"tests": {"passed": npass, "failed": nfail, "failures": [n for n, ok in RESULTS if not ok]}, "mutants": {"killed": nkill, "survived": nsurv, "survivors": [n for n, ok in MUTANTS if not ok]},
           "groups": {"A_asserter_mutants": len(MA), "K_run_record_mutants": len(MK), "K_reach_mutants": len(MR)}}, open(HERE / "selftest_chain_result.json", "w", encoding="utf-8"), indent=1)
sys.exit(0 if nfail == 0 and nsurv == 0 else 1)
