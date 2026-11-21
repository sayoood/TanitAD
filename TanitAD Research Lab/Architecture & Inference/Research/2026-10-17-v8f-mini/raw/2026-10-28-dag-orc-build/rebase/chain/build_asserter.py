#!/usr/bin/env python3
"""Assemble box2_ship/gate_dagoaccorc_assert.py (DAG-ORC's gate asserter re-based on DAG-OACC, Amendment 1).

    python build_asserter.py     writes ./box2_ship/gate_dagoaccorc_assert.py (LF)

ADAPT, DO NOT REWRITE: the functions that already exist and were proven are EXTRACTED BYTE FOR BYTE from the installed files by AST segment, never retyped:
  from C:/Users/Admin/box2_ship/gate_oaf_assert_r2.py (DAG-OACC's r2 asserter):  finite, wacc_problems (read_o_grad_diff keyed on residual_zero_init), swap_problems (the DAgger swap reached; the bank; K-B2 / K-B3 / COVERAGE)
  from E:/Projects/TanitAD-artifacts/dag_orc_chain/box2_ship/gate_dagorc_assert.py (DAG-ORC's asserter, DAG-based): argv_of, flags, argv_sha256, drop
The NEW code (ego_problems, expected_from_ref, check, write_token, main) is the composition of the two: DAG-OACC's recipe verbatim + the perturbation composed with the swap.
"""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
R2 = Path("C:/Users/Admin/box2_ship/gate_oaf_assert_r2.py")
ORC = Path("E:/Projects/TanitAD-artifacts/dag_orc_chain/box2_ship/gate_dagorc_assert.py")


def seg(path, names):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out[node.name] = ast.get_source_segment(src, node)
    assert set(out) == set(names), (path, set(names) - set(out))
    return out


r2 = seg(R2, ["finite", "wacc_problems", "swap_problems"])
orc = seg(ORC, ["argv_of", "flags", "argv_sha256", "drop"])
assert "ARMS[arm]" in r2["swap_problems"]      # patched below to SWAP_CFG[arm]
r2["swap_problems"] = r2["swap_problems"].replace("ARMS[arm]", "SWAP_CFG[arm]")

HEAD = '''#!/usr/bin/env python3
"""PREREG_V8F_DAG_ORC on box 2, RE-BASED ON DAG-OACC by Amendment 1 (A1-2026-10-27): the assertions the chain makes on a PULLED launch-gate selfcheck.json, and the launch-gate token that exists only after they pass.
Derived from gate_dagorc_assert.py (DAG-based) and gate_oaf_assert_r2.py (DAG-OACC's r2): finite / wacc_problems / swap_problems are DAG-OACC's r2 functions BYTE FOR BYTE (extracted by build_asserter.py), argv_of / flags /
argv_sha256 / drop are DAG-ORC's; the rest is the composition.

    gate_dagoaccorc_assert.py check --selfcheck SC --gate-line F --train-line F --pin-line F --arm dagoaccorc|dagoaccorcrep --seed S --banked-init-sha M2B_SHA --bank-sha BANK_SHA --bank-json BANK.json --w-acc 49.6969
                              [--b-summary DAG-OACC summary.json] [--ref-argv-json DAG-OACC argv.json] [--expect expect.json]
    gate_dagoaccorc_assert.py token (the same arguments) --commit C40 --queue-sha SHA --key-sha-json JSON --run NAME --out token.json

The two arms (Amendment 1 section 1), against DAG-OACC_s's BANKED argv (``--ref-argv-json``, the one-lever reference: ``v8f-dagoacc/M2b_s<S>_dagoacc/argv.json``):
  dagoaccorc     (A)   the reference with ``--ego-perturb-p 0`` replaced by ``--ego-perturb-p 0.3`` + the five law flags right after it (vscale 0.5 1.0, vmax 8.0, adrop 1.5, yawrate-std 2.0, seed = SEED), and the bare flag
                       ``--ego-perturb-with-dagger`` appended at the END; only --out differs otherwise.  Gate: the SWAP reaches the loss (dagger_probe, DAG-OACC's), the PERTURBATION reaches the loss (ego_perturb_probe, every control
                       exactly 0.0, the p = 0 control included), the acc term reaches O's head (acc_probe: read_o_grad_diff finite and > 0, because the init is the TRAINED M2b), the composition flag is BUILT.
  dagoaccorcrep  (REP) the reference with ``--seed`` = 10 + SEED; only --out and --seed differ; no perturbation (the ego probe reads 'off'), the swap on, the flag absent.
B (p 0.1) is NOT queued (Amendment 1 section 4) and has no arm here.  ``--gate-line`` / ``--train-line`` = the lines the QUEUE PRINTS under V8F_DRYRUN; ``--pin-line`` = the TRAINING line box 2 itself pinned.  Exit 0 = every
assertion holds; 1 = the list of failures is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys

REAL = {"steps": 3000, "warmup": 100, "n_clips": 4512, "n_excluded": 60, "holdout_sha": "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9", "windows": 306566, "hist": 4, "hmax": 16, "window_start": 3,
        "init_tail": "/M2b_s{seed}/ckpt_last.pt"}
SECOND_LEVER = ("--hist", "--hmax", "--window-start", "--w-tac", "--w-cons", "--w-traj", "--w-pen", "--w-pen-self", "--w-wm", "--code-mix", "--data-frac", "--max-clips", "--cons-absent", "--wm", "--navonly",
                "--no-consistency", "--detach-upper", "--no-resume", "--o-interface")      # (--w-acc is the BASE recipe's, asserted equal to W; --o-interface is absent from DAG-OACC's argv)
EGO_LAW = ("--ego-perturb-vscale", "--ego-perturb-vmax", "--ego-perturb-adrop", "--ego-perturb-yawrate-std", "--ego-perturb-seed")
COMPOSE = "--ego-perturb-with-dagger"
ARMS = {"dagoaccorc": 0.3, "dagoaccorcrep": 0.0}      # arm -> the registered ego p (Amendment 1 section 1)
SWAP_P = 0.1
SWAP_CFG = {a: {"p": True, "init_tail": "/M2b_s{seed}/ckpt_last.pt", "bank_tail": "/dagger_bank/s{seed}/bank.json", "o_interface": "full", "init_name": "M2b"} for a in ARMS}      # swap_problems' per-arm config (DAG-OACC's)
LAW_VALUES = {"ego_perturb_vscale": [0.5, 1.0], "ego_perturb_vmax": 8.0, "ego_perturb_adrop": 1.5, "ego_perturb_yawrate_std": 2.0}      # + ego_perturb_seed == SEED
LAW_DEFAULT = {"ego_perturb_vscale": [0.5, 1.0], "ego_perturb_vmax": 8.0, "ego_perturb_adrop": 1.5, "ego_perturb_yawrate_std": 0.0, "ego_perturb_seed": 0}      # the trainer's defaults, built when no law flag is named (p = 0)
PROBE_CONTROLS = ("control_repeat_loss_diff", "control_repeat_grad_diff", "control_p0_loss_diff", "control_p0_grad_diff", "control_prior_grad_diff")
'''

NEW = '''

def ego_problems(sc: dict, arm: str, p: float, seed: int) -> list:
    """The perturbation lever (A): G-LIVE on a PERTURBED batch, the perturbation REACHES the loss the trainer backpropagates, every control reads EXACTLY 0.0 (the p = 0 control included); the built record is the registered one;
    the composition flag is BUILT.  REP: the probe reads 'off', nothing is applied, the flag is not built."""
    bad = []
    lv = sc["levers"]
    pr = sc.get("ego_perturb_probe")
    if not isinstance(pr, dict) or pr.get("pass") is not True:
        bad.append(f"ego_perturb_probe {pr}")
    elif p > 0.0:
        if pr.get("ego_perturb") == "off" or pr.get("p_declared") != p:
            bad.append(f"the ego probe ran for p {pr.get('p_declared')!r} ({pr.get('ego_perturb')!r}), the run's is {p}")
        if not (isinstance(pr.get("n_selected_at_p1"), int) and pr["n_selected_at_p1"] > 0):
            bad.append(f"the ego probe saw no eligible window ({pr.get('n_selected_at_p1')})")
        if not (pr.get("loss_diff", 0.0) != 0.0 and pr.get("grad_diff_all", 0.0) > 0.0):
            bad.append(f"the perturbed batch moves the loss by {pr.get('loss_diff')} and the gradient by {pr.get('grad_diff_all')}: the perturbation does not reach the loss")
        for k in PROBE_CONTROLS:
            if pr.get(k) != 0.0:
                bad.append(f"ego probe control {k} = {pr.get(k)!r} (must be exactly 0.0)")
        if pr.get("control_p0_same_object") is not True:
            bad.append("ego probe: p = 0 is not the identity (the batch object changed)")
        if pr.get("control_wm_grad_diff") not in (None, 0.0):
            bad.append("the world-model auxiliary's gradient moved with the perturbation")
        lp = pr.get("live_perturbed") or {}
        if lp.get("pass") is not True or lp.get("control_prior_grad_sum") != 0.0:
            bad.append(f"G-LIVE on the PERTURBED batch: pass {lp.get('pass')} / prior control {lp.get('control_prior_grad_sum')}")
        if pr.get("ego_dagger_overlap_n") not in (None, 0):
            bad.append(f"the composed probe batch has ego_dagger_overlap_n {pr.get('ego_dagger_overlap_n')!r} (must be exactly 0 when recorded)")
    elif pr.get("ego_perturb") != "off":
        bad.append(f"REP (p = 0): the ego probe should read 'off' (nothing is applied), it read {pr.get('ego_perturb')!r}")
    ep = lv.get("ego_perturb") or {}
    want_ep = {"ego_perturb_p": p, **({**LAW_VALUES, "ego_perturb_seed": seed} if p > 0.0 else LAW_DEFAULT), "kin_mode": "yaw"}
    for k, w in want_ep.items():
        if ep.get(k) != w:
            bad.append(f"built ego_perturb {k} {ep.get(k)!r} != {w!r}")
    wd = lv.get("ego_perturb_with_dagger")
    if p > 0.0 and wd is not True:
        bad.append(f"built ego_perturb_with_dagger {wd!r} on A (must be True: the composition is the arm)")
    if p == 0.0 and wd not in (False, None):
        bad.append(f"built ego_perturb_with_dagger {wd!r} on REP (the flag must be absent)")
    return bad


def expected_from_ref(ref: list, arm: str, seed: int) -> list:
    """The registered argv of ``arm`` built from DAG-OACC's banked argv (``ref``, ``--out`` dropped): A = ``--ego-perturb-p 0`` -> ``--ego-perturb-p 0.3`` + the five law flags right after it, and the bare
    ``--ego-perturb-with-dagger`` appended at the END; REP = ``--seed`` -> 10 + SEED.  Raises ValueError when the reference does not have the shape the rule needs."""
    r = drop(ref)
    if r.count("--ego-perturb-p") != 1 or r.count("--seed") != 1:
        raise ValueError("the reference argv must name --ego-perturb-p and --seed exactly once")
    if COMPOSE in r or any(k in r for k in EGO_LAW):
        raise ValueError("the reference (DAG-OACC) argv must not carry the composition flag or any law flag")
    kp, ks = r.index("--ego-perturb-p"), r.index("--seed")
    if r[kp + 1] != "0" or r[ks + 1] != str(seed):
        raise ValueError(f"the reference argv must carry --ego-perturb-p 0 and --seed {seed}, it has {r[kp + 1]!r} and {r[ks + 1]!r}")
    if arm == "dagoaccorcrep":
        out = list(r)
        out[ks + 1] = str(10 + seed)
        return out
    law = ["--ego-perturb-vscale", "0.5", "1.0", "--ego-perturb-vmax", "8.0", "--ego-perturb-adrop", "1.5", "--ego-perturb-yawrate-std", "2.0", "--ego-perturb-seed", str(seed)]
    return r[:kp + 1] + ["0.3"] + law + r[kp + 2:] + [COMPOSE]


def check(a) -> list:
    sc = json.load(open(a.selfcheck, encoding="utf-8"))
    exp = {**REAL, **(json.load(open(a.expect, encoding="utf-8")) if a.expect else {})}
    seed, arm = int(a.seed), a.arm
    p = ARMS[arm]
    tseed = 10 + seed if arm == "dagoaccorcrep" else seed      # the trainer's --seed
    bad = []
    i, lv = sc["info"], sc["levers"]
    gl, tl = argv_of(open(a.gate_line, encoding="utf-8").read()), argv_of(open(a.train_line, encoding="utf-8").read())
    pin = argv_of(open(a.pin_line, encoding="utf-8").read())
    # ---- THE PASS TOKEN's ingredients: the trainer's in-process gate wrote these
    if sc.get("declared_vs_built") != "PASS":
        bad.append("declared_vs_built")
    live = sc["live"]
    if not live.get("pass") or live.get("control_prior_grad_sum") != 0.0:
        bad.append(f"live {live.get('pass')} / prior control {live.get('control_prior_grad_sum')}")
    if live.get("dead_params") != [] or live.get("declared_dead") != []:
        bad.append(f"G-LIVE dead set {live.get('dead_params')} (declared {live.get('declared_dead')}) != []: the fine-tune kills no weight")
    # ---- THE LEVERS: the perturbation (A) and its composition with the swap; the swap (both arms, DAG-OACC's); the acceleration term (both arms, DAG-OACC's)
    bad += ego_problems(sc, arm, p, seed)
    bad += swap_problems(sc, arm, SWAP_P, seed, a)
    try:
        W = float(a.w_acc)
    except (TypeError, ValueError):
        W = float("nan")
    if not (finite(W) and W > 0.0):
        bad.append(f"--w-acc {a.w_acc!r} is not a finite positive weight")
    else:
        bad += wacc_problems(sc, W)
    if lv.get("o_interface", "full") != "full":
        bad.append(f"built o_interface {lv.get('o_interface')!r} != 'full' (DAG-OACC's)")
    if lv.get("code_mix", "whole") != "whole":
        bad.append(f"built code_mix {lv.get('code_mix')!r} != 'whole'")
    # ---- THE START STATE: M2b_s<seed>'s checkpoint (DAG-OACC's init, from its launch-gate token), hashed and read back (K-INIT)
    ic = lv.get("init_ckpt") or {}
    if ic.get("sha256") != a.banked_init_sha:
        bad.append(f"the fine-tune does not start from M2b_s{seed}'s checkpoint: init sha256 {str(ic.get('sha256'))[:16]} != {a.banked_init_sha[:16]}")
    if i.get("init_ckpt") != ic:
        bad.append("info.init_ckpt and levers.init_ckpt disagree")
    if ic.get("applied") is not True or ic.get("n_mismatch_after_load") != 0 or not ic.get("n_tensors"):
        bad.append(f"init_ckpt not applied / read back: applied {ic.get('applied')} mismatches {ic.get('n_mismatch_after_load')} tensors {ic.get('n_tensors')}")
    itail = exp["init_tail"].format(seed=seed)
    if not str(ic.get("path", "")).replace("\\\\", "/").endswith(itail):
        bad.append(f"init_ckpt path {ic.get('path')!r} is not ...{itail}")
    # ---- THE FINE-TUNE'S SCHEDULE (DAG-OACC's = DAG's)
    steps = exp["steps"]
    if sc.get("steps_total") != steps:
        bad.append(f"steps {sc.get('steps_total')} != the PREREG's {steps}")
    sch = lv.get("lr_schedule") or {}
    if (sch.get("lr"), sch.get("warmup"), sch.get("total")) != (1e-4, exp["warmup"], steps):
        bad.append(f"built lr schedule {sch} != lr 1e-4 / warmup 100 / total {steps}")
    if (lv.get("p_gt_start"), lv.get("p_gt_end")) != (0.2, 0.2):
        bad.append(f"built p_gt {lv.get('p_gt_start')} -> {lv.get('p_gt_end')}, expected 0.2 -> 0.2")
    # ---- the comparator "B" is DAG-OACC's banked summary.json: same data, same loss weights, same parameter groups, same w_acc
    if (i["n_clips"], i["n_excluded"], i["n_excluded_declared"], i["excluded_in_train"]) != (exp["n_clips"], exp["n_excluded"], exp["n_excluded"], 0):
        bad.append("clips / holdout counts")
    if i["excluded_sha256"] != exp["holdout_sha"]:
        bad.append("the excluded set is not the RA-1 holdout")
    if i["kin_mode"] != "yaw":
        bad.append("kin_mode")
    if (lv["wm"], lv["navonly"], lv["consistency"], lv.get("cons_absent")) != ("b", False, True, "mask"):
        bad.append(f"levers wm / navonly / consistency / cons_absent {(lv['wm'], lv['navonly'], lv['consistency'], lv.get('cons_absent'))}")
    if (lv.get("hist"), lv.get("hmax"), lv.get("window_start")) != (exp["hist"], exp["hmax"], exp["window_start"]):
        bad.append(f"built hist / hmax / window_start {(lv.get('hist'), lv.get('hmax'), lv.get('window_start'))} != {(exp['hist'], exp['hmax'], exp['window_start'])}")
    if (i.get("window_start"), i.get("window_t_min"), i.get("ds_window_start")) != (exp["window_start"],) * 3:
        bad.append(f"window start (asked / first t / dataset clamp origin) {(i.get('window_start'), i.get('window_t_min'), i.get('ds_window_start'))} != ({exp['window_start']},) * 3")
    if i.get("n_windows_all_clips") != exp["windows"]:
        bad.append(f"n_windows_all_clips {i.get('n_windows_all_clips')} != the PREREG's {exp['windows']}")
    if a.b_summary:
        b = json.load(open(a.b_summary, encoding="utf-8"))
        bl, wd = b["levers"], lv.get("w") or {}
        if i["train_sha256"] != b["train_sha256"]:
            bad.append(f"TRAIN list {i['train_sha256'][:12]} != B's {b['train_sha256'][:12]}")
        for k in ("w_pen", "w_pen_self", "w_wm"):
            if wd.get(k) != bl["w"][k]:
                bad.append(f"{k} {wd.get(k)} != B's {bl['w'][k]}")
        if (wd.get("w_tac"), wd.get("w_cons")) != (1.0, 0.5):
            bad.append(f"w_tac / w_cons {wd.get('w_tac')} / {wd.get('w_cons')} != B's 1.0 / 0.5 (a second lever moved)")
        if lv.get("w_acc") != bl.get("w_acc"):
            bad.append(f"built w_acc {lv.get('w_acc')!r} != DAG-OACC's banked {bl.get('w_acc')!r}")
        if i.get("n_windows_all_clips") != b.get("n_windows"):
            bad.append(f"the window set is not B's: n_windows_all_clips {i.get('n_windows_all_clips')} != B's {b.get('n_windows')}")
        pg, pb = lv["params"], bl["params"]
        if lv["n_params_trainable"] != bl["n_params_trainable"]:
            bad.append(f"n_params_trainable {lv['n_params_trainable']} != B's {bl['n_params_trainable']}")
        for k in sorted(set(pg) | set(pb)):
            if pg.get(k, 0) != pb.get(k, 0):
                bad.append(f"parameter group {k}: {pg.get(k)} vs B's {pb.get(k)}")
    # ---- THE QUEUE'S OWN PRINTED LINES: every lever NAMED exactly once with its registered value (the flags' defaults disable the lever), nothing else moved
    want = {"--arm": ["M2b"], "--seed": [str(tseed)], "--grid": ["pool4x10"], "--steps": [str(steps)], "--lr": ["1e-4"], "--warmup": [str(exp["warmup"])], "--p-gt-start": ["0.2"], "--p-gt-end": ["0.2"],
            "--ego-perturb-p": [{0.3: "0.3", 0.0: "0"}[p]], "--dagger-p": ["0.1"], "--dagger-bank-sha256": [a.bank_sha.lower()], "--dagger-seed": [str(seed)], "--w-acc": [a.w_acc]}
    law_want = {"--ego-perturb-vscale": ["0.5", "1.0"], "--ego-perturb-vmax": ["8.0"], "--ego-perturb-adrop": ["1.5"], "--ego-perturb-yawrate-std": ["2.0"], "--ego-perturb-seed": [str(seed)]}
    if p > 0.0:
        want.update(law_want)
    for name, line in (("train", tl), ("gate", gl), ("pin", pin)):
        fl = flags(line)
        for k, w in want.items():
            if fl.get(k) != [w]:
                bad.append(f"the {name} line does not carry {k} {' '.join(w)} exactly once (it has {fl.get(k)})")
        ik = fl.get("--init-ckpt") or []
        if len(ik) != 1 or len(ik[0]) != 1 or not ik[0][0].replace("\\\\", "/").endswith(itail):
            bad.append(f"the {name} line --init-ckpt {ik} is not ...{itail} exactly once")
        if len(fl.get("--dagger-bank") or []) != 1:
            bad.append(f"the {name} line does not name --dagger-bank exactly once (it has {fl.get('--dagger-bank')})")
        if p == 0.0:
            for k in EGO_LAW:
                if k in fl:
                    bad.append(f"the {name} line names {k} on REP: the trainer's gate refuses a law flag set while p is 0 (REP names ONLY --ego-perturb-p 0)")
            if COMPOSE in fl:
                bad.append(f"the {name} line names {COMPOSE} on REP: the flag must be absent")
        elif fl.get(COMPOSE) != [[]]:
            bad.append(f"the {name} line does not name {COMPOSE} exactly once as a bare flag (it has {fl.get(COMPOSE)})")
        for fx in SECOND_LEVER:
            if fx in fl:
                bad.append(f"the {name} line sets {fx}: a second lever beside the arm's own")
    if gl.count("--selfcheck-only") != 1:
        bad.append("the gate line does not carry --selfcheck-only exactly once")
    if "--selfcheck-only" in tl or "--selfcheck-only" in pin:
        bad.append("the TRAIN / pinned line carries --selfcheck-only")
    if drop([x for x in gl if x != "--selfcheck-only"]) != drop(tl):
        bad.append("the gate argv is not the training argv with --out and --selfcheck-only changed")
    if pin != tl:
        bad.append("THE PIN IS NOT THE PLANNED TRAINING ARGV: box 2 pinned (and will run) a line other than the one the token binds")
    if sc.get("argv_sha") != hashlib.sha256(" ".join(gl).encode()).hexdigest()[:16]:
        bad.append("the gate ran an argv other than the queue's printed gate line (the selfcheck's argv_sha differs)")
    # ---- THE ONE-LEVER RULE (K-PAIR) against DAG-OACC's banked argv: A = ref + the perturbation + the flag exactly; REP = ref with --seed only; only --out differs
    onelever = "skipped (no --ref-argv-json: a synthetic world has no banked reference)"
    if a.ref_argv_json:
        ref = json.load(open(a.ref_argv_json, encoding="utf-8"))["argv"]
        try:
            exp_argv = expected_from_ref(ref, arm, seed)
        except ValueError as e:
            bad.append(f"ONE-LEVER: {e}")
            exp_argv = None
        onelever = "ok"
        if exp_argv is not None:
            for name, line in (("train", tl), ("gate", [x for x in gl if x != "--selfcheck-only"]), ("pin", pin)):
                if drop(line) != exp_argv:
                    k = next((k for k in range(min(len(exp_argv), len(drop(line)))) if exp_argv[k] != drop(line)[k]), min(len(exp_argv), len(drop(line))))
                    bad.append(f"ONE-LEVER: the {name} line is not DAG-OACC's banked argv + exactly the registered change for {arm} (first difference at token {k}: {drop(line)[k:k + 2]} vs {exp_argv[k:k + 2]}; only --out may differ)")
                    onelever = "NOT"
    if not i.get("lru"):
        bad.append("info.lru is not recorded in the selfcheck (the pipeline sizing must be on the record)")
    print("DAG-ORC (DAG-OACC base) gate:", "PASS" if not bad else bad, "|", arm, "s%d" % seed, "| ego p", p, "| trainer seed", tseed, "| clips", i["n_clips"], "steps", sc.get("steps_total"), "windows", i.get("n_windows_all_clips"),
          "| init", str(ic.get("sha256"))[:12], "| swap probe grad diff", (sc.get("dagger_probe") or {}).get("grad_diff_all"), "| ego probe grad diff", (sc.get("ego_perturb_probe") or {}).get("grad_diff_all"),
          "| params", lv["n_params_trainable"], "| lru", i.get("lru"), "| one-lever:", onelever)
    return bad


def write_token(a) -> int:
    sc = json.load(open(a.selfcheck, encoding="utf-8"))
    g_argv, t_argv = argv_of(open(a.gate_line, encoding="utf-8").read()), argv_of(open(a.pin_line, encoding="utf-8").read())
    bad = []
    if sc.get("declared_vs_built") != "PASS":
        bad.append("the gate's selfcheck is not a PASS")
    if sc.get("argv_sha") != hashlib.sha256(" ".join(g_argv).encode()).hexdigest()[:16]:
        bad.append("the gate ran an argv other than the queue's printed gate line (the selfcheck's argv_sha differs)")
    if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
        bad.append(f"commit {a.commit!r} is not 40-hex")
    data = sc["info"]["train_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", data):
        bad.append("data_sha256 is not 64-hex")
    if "--selfcheck-only" in t_argv:
        bad.append("the TRAINING argv carries --selfcheck-only")
    if bad:
        print("no token:", bad)
        return 1
    tok = {"kind": "launch-gate-token", "verdict": "PASS", "prereg": "PREREG_V8F_DAG_ORC", "amendment": "A1-2026-10-27 (re-based on DAG-OACC)", "run": a.run,
           "binding": {"commit": a.commit, "argv_sha256": argv_sha256(t_argv), "data_sha256": data},
           "binding_note": "argv_sha256 = launch_gate.argv_sha256 of the TRAINING argv as PINNED BY BOX 2 at gate time (the file traindag executes), which equals the queue's planned training line (asserted); argv.json will record it",
           "gate": {"kind": "v8f_mini_train.py --selfcheck-only (in-process: G-DVB FLAG_CHECKS + bad_ego_perturb + bad_dagger + bad_init_ckpt + G-LIVE + ego_perturb_probe + dagger_probe + acc_probe)", "argv_sha256": argv_sha256(g_argv), "argv_sha16": sc.get("argv_sha"),
                    "selfcheck_sha256": hashlib.sha256(open(a.selfcheck, "rb").read()).hexdigest()},
           "queue_sha256": a.queue_sha, "key_files_sha256": json.loads(a.key_sha_json), "excluded_sha256": sc["info"]["excluded_sha256"], "init_ckpt_sha256": sc["levers"]["init_ckpt"]["sha256"],
           "bank_sha256": a.bank_sha.lower(), "dagger_p": SWAP_P, "w_acc": float(a.w_acc), "o_interface": "full",
           "arm": a.arm, "ego_perturb_p": ARMS[a.arm], "ego_perturb_with_dagger": ARMS[a.arm] > 0.0, "trainer_seed": 10 + int(a.seed) if a.arm == "dagoaccorcrep" else int(a.seed),
           "v9_md5": {"v9_labels_train.npz": "83e30bd3791f0767707eade5f4671759", "v9_labels_eval139.npz": "6b5c7f207cffc3b7eb3cd527fd433599"},
           "hmac": "none: written by the box-2 chain, not by launch_gate.py (its key lives on the launch host); the reader checks the binding and does not verify an HMAC"}
    json.dump(tok, open(a.out, "w", encoding="utf-8"), indent=1)
    print("gate token written for", a.run, "commit", a.commit, "argv_sha256", tok["binding"]["argv_sha256"][:16])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("check", "token"))
    ap.add_argument("--selfcheck", required=True)
    ap.add_argument("--gate-line", required=True)
    ap.add_argument("--train-line", required=True)
    ap.add_argument("--pin-line", required=True)
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--banked-init-sha", required=True)
    ap.add_argument("--bank-sha", required=True)
    ap.add_argument("--bank-json", required=True)
    ap.add_argument("--w-acc", required=True)
    ap.add_argument("--b-summary", default="")
    ap.add_argument("--ref-argv-json", default="")
    ap.add_argument("--expect", default="")
    ap.add_argument("--commit", default="")
    ap.add_argument("--queue-sha", default="")
    ap.add_argument("--key-sha-json", default="{}")
    ap.add_argument("--run", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.seed not in ("0", "1"):
        print("REFUSED: --seed must be 0 or 1 (the seed of M2b_s, the init)")
        return 2
    bad = check(a)
    if bad:
        return 1
    return write_token(a) if a.mode == "token" else 0


if __name__ == "__main__":
    sys.exit(main())
'''

parts = [HEAD.rstrip("\n"), "", "", orc["argv_of"], "", "", orc["flags"], "", "", orc["argv_sha256"], "", "", orc["drop"], "", "", r2["finite"], "", "", r2["wacc_problems"], "", "", r2["swap_problems"], NEW]
text = "\n".join(parts).replace("\r", "")
# the doubled backslashes in NEW's string literals were written for this build file's non-raw ''' string: collapse to the single-escaped form the asserter needs
(HERE / "box2_ship").mkdir(exist_ok=True)
out = HERE / "box2_ship" / "gate_dagoaccorc_assert.py"
out.write_text(text, encoding="utf-8", newline="\n")
import py_compile
py_compile.compile(str(out), doraise=True)
print(f"wrote {out} sha256 {hashlib.sha256(text.encode()).hexdigest()} ({len(text.splitlines())} lines)")
