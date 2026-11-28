#!/usr/bin/env python3
"""PREREG_V8F_DAG_OACC and PREREG_V8F_FLATDAG on box 2 (the DAG2 build's gate_dag2_assert.py, generalised): the assertions the chain makes on a PULLED launch-gate selfcheck.json, and the launch-gate token that exists
only after they pass.

r2 (2026-10-27): the ONLY change against gate_oaf_assert.py is in ``wacc_problems``: ``acc_probe.read_o_grad_diff`` is NOT a registered control. It read exactly 0.0 in OACC's gate only because that gate was FROM SCRATCH with O's
residual head zero-initialised (``levers.residual_zero_init`` True: through a zero last layer ``read_o`` gets exactly 0 gradient). DAG-OACC fine-tunes the TRAINED M2b checkpoint (``residual_zero_init`` False), so ``read_o``
MUST receive the term's gradient (box 2 read 864.79 on s0). r2: ``residual_zero_init`` True -> read_o_grad_diff exactly 0.0 (the zero-layer identity); False -> finite and > 0; anything else -> not decidable, refused.
The four registered controls stay "exactly 0.0". flatdag / flatdag0 logic and every other function are byte-identical to gate_oaf_assert.py (selftest_oaf_r2.py R-section proves it on the AST).

Arms:

  dagoacc   DAG's fine-tune argv VERBATIM (init M2b_s<seed>, the ITERATION-1 bank, --dagger-p 0.1) + `--w-acc W`, nothing else (C-O1). Asserted: the acc term finite non-zero in the gate (acc_probe, live.terms), head_o's gradient from it
            non-zero, the swap reached (dagger_probe), built w_acc == argv's == the chain's W, and the ONE-LEVER rule on the argv text: the pinned dagoacc line == box 2's DAG pin + `--w-acc W`, only --out differing.
  flatdag   FLAT_s<seed> (--o-interface cut KEPT, named) fine-tuned with DAG's recipe, --dagger-p 0.1, the FLAT bank. Asserted: BOTH the cut (levers.o_interface 'cut', the trainer's o_interface_probe: traj60 invariant to the
            donor / GT / random code and the shifted checkpoints, O handed constants; G-LIVE's dead set is exactly ['e_code.weight']) AND the swap reached (dagger_probe PASS in the same gate), and FLAT's every other flag.
  flatdag0  the same argv with --dagger-p 0 and no bank / sha / seed flag (C-F2); the cut is asserted, the swap probe must read 'off'.

    gate_oaf_assert.py check|token --selfcheck SC --gate-line F --train-line F --pin-line F --arm dagoacc|flatdag|flatdag0 --seed S [--p P] [--bank-sha SHA --bank-json bank.json] --banked-init-sha SHA
                       [--w-acc W --dag-pin-line F] [--ref-argv-json F] [--b-summary B.json] [--expect expect.json] [--commit C40 --queue-sha SHA --key-sha-json JSON --run NAME --out token.json]

``--ref-argv-json``: dagoacc = DAG_s<seed>'s banked argv.json (its `argv` must equal box 2's DAG pin, and the dagoacc pin must be that + --w-acc W); flat arms = FLAT_s<seed>'s banked argv.json (every flag but the fine-tune
and bank flags must be FLAT's). Exit 0 = every assertion holds; 1 = the list of failures is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys

REAL = {"steps": 3000, "n_clips": 4512, "n_excluded": 60, "holdout_sha": "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9", "windows": 306566, "hist": 4, "hmax": 16, "window_start": 3, "check_b_levers": True}
ARMS = {
    "dagoacc": {"prereg": "PREREG_V8F_DAG_OACC", "p": True, "init_tail": "/M2b_s{seed}/ckpt_last.pt", "bank_tail": "/dagger_bank/s{seed}/bank.json", "o_interface": "full", "init_name": "M2b"},
    "flatdag": {"prereg": "PREREG_V8F_FLATDAG", "p": True, "init_tail": "/M2b_s{seed}_flat/ckpt_last.pt", "bank_tail": "/dagger_bank_flat/s{seed}/bank.json", "o_interface": "cut", "init_name": "FLAT"},
    "flatdag0": {"prereg": "PREREG_V8F_FLATDAG", "p": False, "init_tail": "/M2b_s{seed}_flat/ckpt_last.pt", "bank_tail": None, "o_interface": "cut", "init_name": "FLAT"},
}
SECOND_LEVER = ("--hist", "--hmax", "--window-start", "--w-tac", "--w-cons", "--w-traj", "--w-pen", "--w-pen-self", "--w-wm", "--code-mix", "--data-frac", "--max-clips", "--cons-absent", "--wm", "--navonly",
                "--no-consistency", "--detach-upper", "--no-resume")
EGO_LAW = ("--ego-perturb-vscale", "--ego-perturb-vmax", "--ego-perturb-adrop", "--ego-perturb-yawrate-std", "--ego-perturb-seed")
DAG_ONLY = ("--dagger-bank", "--dagger-bank-sha256", "--dagger-seed")
FINE_TUNE = {"--out", "--init-ckpt", "--steps", "--lr", "--warmup", "--p-gt-start", "--p-gt-end", "--ego-perturb-p", "--dagger-p", "--dagger-bank", "--dagger-bank-sha256", "--dagger-seed"}


def argv_of(text: str) -> list:
    t = [x.strip('"') for x in re.findall(r'"[^"]*"|\S+', text.strip())]
    return t[next(k for k, x in enumerate(t) if x.lower().endswith("v8f_mini_train.py")) + 1:]


def flags(a: list) -> dict:
    o, k = {}, 0
    while k < len(a):
        if a[k].startswith("--"):
            j = k + 1
            while j < len(a) and not a[j].startswith("--"):
                j += 1
            o.setdefault(a[k], []).append(a[k + 1:j])
            k = j
        else:
            k += 1
    return o


def argv_sha256(argv) -> str:
    return hashlib.sha256(json.dumps(list(argv), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def drop(x: list, names=("--out",), bare=()) -> list:
    o, k = [], 0
    while k < len(x):
        if x[k] in names:
            k += 2
        elif x[k] in bare:
            k += 1
        else:
            o.append(x[k])
            k += 1
    return o


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def wacc_problems(sc: dict, W: float) -> list:
    """C-O2 on the gate's record: the acceleration term runs (finite, non-zero), the built weight is the argv's, O's head gets its gradient, and the controls read exactly 0.0."""
    bad = []
    lv, live, ap = sc["levers"], sc["live"], sc.get("acc_probe")
    if lv.get("w_acc") != W:
        bad.append(f"built levers.w_acc {lv.get('w_acc')!r} != the argv's {W!r}")
    if "acc" not in (live.get("terms") or {}) or not finite(live["terms"]["acc"]) or live["terms"]["acc"] == 0.0:
        bad.append(f"G-LIVE: the acc term is {(live.get('terms') or {}).get('acc')!r} (must be in the loss dict, finite and non-zero)")
    if live.get("missing_terms") not in ([], None):
        bad.append(f"G-LIVE missing_terms {live.get('missing_terms')}")
    if not isinstance(ap, dict) or ap.get("pass") is not True:
        return bad + [f"acc_probe {ap}"]
    if ap.get("w_acc_declared") != W or ap.get("w_probe") != W:
        bad.append(f"acc_probe declared / probe weight {ap.get('w_acc_declared')!r} / {ap.get('w_probe')!r} != {W!r}")
    if not (finite(ap.get("acc")) and ap["acc"] > 0.0):
        bad.append(f"acc_probe.acc {ap.get('acc')!r} is not finite and non-zero")
    if not (finite(ap.get("contribution")) and finite(ap.get("w_x_acc")) and abs(ap["contribution"] - ap["w_x_acc"]) <= 1e-4 * max(1.0, abs(ap["w_x_acc"])) and ap["contribution"] > 0.0):
        bad.append(f"the total moved by {ap.get('contribution')!r} for w x acc = {ap.get('w_x_acc')!r}: the term does not enter the loss at the declared weight")
    if not (finite(ap.get("head_o_grad_diff")) and ap["head_o_grad_diff"] > 0.0):
        bad.append(f"O's head receives gradient {ap.get('head_o_grad_diff')!r} from the acceleration term (must be > 0)")
    if not (finite(live.get("head_o_grad_step0")) and live["head_o_grad_step0"] > 0.0):
        bad.append(f"live.head_o_grad_step0 {live.get('head_o_grad_step0')!r}")
    for k in ("control_repeat_total_diff", "control_repeat_grad_diff", "control_prior_grad_diff", "control_wm_grad_diff"):
        if ap.get(k) != 0.0:
            bad.append(f"acc_probe control {k} = {ap.get(k)!r} (must be exactly 0.0)")
    # read_o_grad_diff is not a control: it is the gradient the term sends THROUGH O's residual head into read_o (traj = prior + basis . head_o(read_o(...)), no detach). Which value is right depends on the start state:
    #   residual_zero_init True  (from scratch: head_o's last layer is exactly zero) -> the identity: exactly 0.0
    #   residual_zero_init False (a TRAINED head, e.g. the fine-tune of M2b)         -> the term must reach read_o: finite and > 0
    rd, zi = ap.get("read_o_grad_diff"), lv.get("residual_zero_init")
    if zi is True:
        if not (finite(rd) and rd == 0.0):
            bad.append(f"acc_probe read_o_grad_diff = {rd!r} (levers.residual_zero_init is True: O's last layer is zero, so read_o gets exactly 0.0)")
    elif zi is False:
        if not (finite(rd) and rd > 0.0):
            bad.append(f"acc_probe read_o_grad_diff = {rd!r} (levers.residual_zero_init is False: a trained head passes the term's gradient to read_o, it must be finite and > 0)")
    else:
        bad.append(f"levers.residual_zero_init = {zi!r} is neither True nor False: the read_o_grad_diff reading {rd!r} cannot be judged")
    return bad


def cut_problems(sc: dict) -> list:
    """C-F3's build half, in the gate: the cut is BUILT (declared == built), O is invariant to everything above it, e_code is the declared dead weight, T still reads S."""
    bad = []
    lv, live, op = sc["levers"], sc["live"], sc.get("o_interface_probe")
    if lv.get("o_interface") != "cut":
        bad.append(f"built o_interface {lv.get('o_interface')!r} != 'cut'")
    if not isinstance(op, dict) or op.get("pass") is not True or op.get("o_interface") != "cut":
        return bad + [f"o_interface_probe {op}"]
    pt = op.get("perturbations") or {}
    for nm in ("donor_code", "gt_code", "random_code", "shifted_checkpoints"):
        q = pt.get(nm)
        if not isinstance(q, dict):
            bad.append(f"o_interface_probe: perturbation {nm} absent")
            continue
        for k in ("max_abs_dtraj60", "o_read_code_abs_max", "o_read_ck_minus_nominal_abs_max", "o_read_ck_ok_abs_max"):
            if q.get(k) != 0.0:
                bad.append(f"o_interface_probe {nm}.{k} = {q.get(k)!r} (the cut: exactly 0.0)")
    sh = pt.get("shifted_checkpoints") or {}
    if not (finite(sh.get("max_abs_d_ckpt")) and sh["max_abs_d_ckpt"] > 0.0 and finite(sh.get("t_reads_s_max_abs_dgoal_logits")) and sh["t_reads_s_max_abs_dgoal_logits"] > 0.0):
        bad.append(f"the checkpoint perturbation did not reach S / T ({sh.get('max_abs_d_ckpt')!r} / {sh.get('t_reads_s_max_abs_dgoal_logits')!r}): the probe is blind")
    if live.get("dead_params") != ["e_code.weight"] or live.get("declared_dead") != ["e_code.weight"]:
        bad.append(f"G-LIVE dead set {live.get('dead_params')} (declared {live.get('declared_dead')}) != ['e_code.weight'] (the cut's dead weight, exactly)")
    return bad


def swap_problems(sc: dict, arm: str, p: float, seed: int, a) -> list:
    bad = []
    i, lv = sc["info"], sc["levers"]
    pr = sc.get("dagger_probe")
    if not isinstance(pr, dict) or pr.get("pass") is not True:
        bad.append(f"dagger_probe {pr}")
    elif ARMS[arm]["p"]:
        if pr.get("dagger") == "off" or pr.get("p_declared") != p:
            bad.append(f"the dagger probe ran for p {pr.get('p_declared')!r} ({pr.get('dagger')!r}), the run's is {p}")
        n_sel = pr.get("n_selected_at_p1")
        if not (isinstance(n_sel, int) and n_sel > 0):
            bad.append(f"the dagger probe saw no eligible window ({n_sel})")
        if not (pr.get("loss_diff", 0.0) != 0.0 and pr.get("grad_diff_all", 0.0) > 0.0):
            bad.append(f"the swapped batch moves the loss by {pr.get('loss_diff')} and the gradient by {pr.get('grad_diff_all')}: the swap does not reach the loss")
        for k in ("control_repeat_loss_diff", "control_repeat_grad_diff", "control_p0_loss_diff", "control_p0_grad_diff", "control_prior_grad_diff"):
            if pr.get(k) != 0.0:
                bad.append(f"dagger probe control {k} = {pr.get(k)!r} (must be exactly 0.0)")
        if pr.get("control_p0_same_object") is not True:
            bad.append("dagger probe: p = 0 is not the identity (the batch object changed)")
        if pr.get("control_wm_grad_diff") not in (None, 0.0):
            bad.append("the world-model auxiliary's gradient moved with the swap")
        if (pr.get("loss_diff_replacing_only") or {}).get("p10f", 0.0) == 0.0:
            bad.append("replacing p10f alone does not move the loss")
        ls = pr.get("live_swapped") or {}
        if ls.get("pass") is not True or ls.get("control_prior_grad_sum") != 0.0:
            bad.append(f"G-LIVE on the SWAPPED batch: pass {ls.get('pass')} / prior control {ls.get('control_prior_grad_sum')}")
        if not (isinstance(pr.get("rc_max_abs_err"), float) and pr["rc_max_abs_err"] <= 1e-5):
            bad.append(f"lab_s.rc frame-change error {pr.get('rc_max_abs_err')!r} (> 1e-5 or absent)")
        if isinstance(n_sel, int) and pr.get("selected_rows_bit_identical_exempt", n_sel) >= n_sel:
            bad.append("every selected probe row is bit-identical to the logged one: the swap cannot be seen")
    elif pr.get("dagger") != "off":
        bad.append(f"p = 0: the dagger probe should read 'off', it read {pr.get('dagger')!r}")
    if lv.get("dagger_p") != p:
        bad.append(f"built dagger_p {lv.get('dagger_p')!r} != {p!r}")
    dg = lv.get("dagger") or {}
    if ARMS[arm]["p"]:
        bsha = a.bank_sha.lower()
        if lv.get("dagger_bank_sha256") != bsha or lv.get("dagger_bank_sha256_declared") != bsha:
            bad.append(f"built / declared bank sha256 {str(lv.get('dagger_bank_sha256'))[:12]} / {str(lv.get('dagger_bank_sha256_declared'))[:12]} != the pulled bank.json's {bsha[:12]}")
        if dg.get("dagger_seed") != seed:
            bad.append(f"built dagger_seed {dg.get('dagger_seed')!r} != {seed}")
        if dg.get("kin_mode") != "yaw":
            bad.append(f"built DAgger kin_mode {dg.get('kin_mode')!r} != 'yaw'")
        bp = str(dg.get("dagger_bank", "")).replace("\\", "/")
        tail = ARMS[arm]["bank_tail"].format(seed=seed)
        if not bp.endswith(tail):
            bad.append(f"built bank path {bp!r} does not end with {tail}")
        d = (i.get("dagger") or {})
        if d.get("bank_sha256") != bsha:
            bad.append(f"info.dagger.bank_sha256 {str(d.get('bank_sha256'))[:12]} != {bsha[:12]}")
        kb2 = d.get("kb2") or {}
        if not (kb2.get("ckpt_sha256") == kb2.get("init_ckpt_sha256") == a.banked_init_sha):
            bad.append(f"K-B2: bank's checkpoint {str(kb2.get('ckpt_sha256'))[:12]} / init {str(kb2.get('init_ckpt_sha256'))[:12]} != {ARMS[arm]['init_name']}_s{seed} {a.banked_init_sha[:12]}")
        cov = d.get("coverage") or {}
        if not (cov.get("n_windows") and cov["n_eligible"] / cov["n_windows"] >= 0.5 and cov.get("blocked") is False):
            bad.append(f"COVERAGE in the gate: {cov}")
        kb3 = d.get("kb3") or {}
        if kb3.get("overlap_with_eval") != 0:
            bad.append(f"K-B3: overlap with eval-141 {kb3.get('overlap_with_eval')!r} != 0")
        bj = json.load(open(a.bank_json, encoding="utf-8"))
        if kb3.get("window_list_sha256") != bj["common"]["window_list_sha256"]:
            bad.append("K-B3: the trainer's TRAIN window-list sha256 is not the one the bank recorded")
    else:
        if lv.get("dagger_bank_sha256") is not None or lv.get("dagger_bank_sha256_declared") is not None or dg.get("dagger_bank") is not None:
            bad.append("p = 0 arm built a bank or declared a bank hash: a lever declared and inert")
        if "dagger" in i:
            bad.append("the p = 0 arm's info carries a DAgger control record (a bank was read at p = 0)")
    return bad


def check(a) -> list:
    sc = json.load(open(a.selfcheck, encoding="utf-8"))
    exp = {**REAL, **(json.load(open(a.expect, encoding="utf-8")) if a.expect else {})}
    seed, arm = int(a.seed), a.arm
    cfg = ARMS[arm]
    p = float(a.p) if cfg["p"] else 0.0
    bad = []
    i, lv = sc["info"], sc["levers"]
    gl, tl = argv_of(open(a.gate_line, encoding="utf-8").read()), argv_of(open(a.train_line, encoding="utf-8").read())
    pin = argv_of(open(a.pin_line, encoding="utf-8").read())
    if sc.get("declared_vs_built") != "PASS":
        bad.append("declared_vs_built")
    live = sc["live"]
    if not live.get("pass") or live.get("control_prior_grad_sum") != 0.0:
        bad.append(f"live {live.get('pass')} / prior control {live.get('control_prior_grad_sum')}")
    if arm == "dagoacc" and (live.get("dead_params") != [] or live.get("declared_dead") != []):
        bad.append(f"G-LIVE dead set {live.get('dead_params')} (declared {live.get('declared_dead')}) != []: the fine-tune kills no weight")
    ep = sc.get("ego_perturb_probe")
    if not isinstance(ep, dict) or ep.get("pass") is not True or ep.get("ego_perturb") != "off":
        bad.append(f"ego_perturb_probe {ep}: DAgger arms never combine with --ego-perturb-p (it must read 'off')")
    if (lv.get("ego_perturb") or {}).get("ego_perturb_p") != 0.0:
        bad.append(f"built ego_perturb_p {(lv.get('ego_perturb') or {}).get('ego_perturb_p')!r} != 0.0")
    # ---- the swap, from the BUILT object
    bad += swap_problems(sc, arm, p, seed, a)
    # ---- the arm's own lever
    if arm == "dagoacc":
        try:
            W = float(a.w_acc)
        except (TypeError, ValueError):
            W = float("nan")
        if not (finite(W) and W > 0.0):
            bad.append(f"--w-acc {a.w_acc!r} is not a finite positive weight")
        else:
            bad += wacc_problems(sc, W)
        if lv.get("o_interface", "full") != "full":
            bad.append(f"built o_interface {lv.get('o_interface')!r} != 'full' (DAG's)")
    else:
        bad += cut_problems(sc)
        if lv.get("w_acc", 0.0) != 0.0:
            bad.append(f"built w_acc {lv.get('w_acc')!r} != 0.0 (FLAT's)")
    if lv.get("code_mix", "whole") != "whole":
        bad.append(f"built code_mix {lv.get('code_mix')!r} != 'whole'")
    # ---- the start state (hashed and read back)
    ic = lv.get("init_ckpt") or {}
    if ic.get("sha256") != a.banked_init_sha:
        bad.append(f"the fine-tune does not start from {cfg['init_name']}_s{seed}'s checkpoint: init sha256 {str(ic.get('sha256'))[:16]} != {a.banked_init_sha[:16]}")
    if i.get("init_ckpt") != ic:
        bad.append("info.init_ckpt and levers.init_ckpt disagree")
    if ic.get("applied") is not True or ic.get("n_mismatch_after_load") != 0 or not ic.get("n_tensors"):
        bad.append(f"init_ckpt not applied / read back: applied {ic.get('applied')} mismatches {ic.get('n_mismatch_after_load')} tensors {ic.get('n_tensors')}")
    itail = cfg["init_tail"].format(seed=seed)
    if not str(ic.get("path", "")).replace("\\", "/").endswith(itail):
        bad.append(f"init_ckpt path {ic.get('path')!r} is not ...{itail}")
    # ---- the fine-tune's schedule (DAG's, unchanged)
    steps = exp["steps"]
    if sc.get("steps_total") != steps:
        bad.append(f"steps {sc.get('steps_total')} != {steps}")
    sch = lv.get("lr_schedule") or {}
    if (sch.get("lr"), sch.get("warmup"), sch.get("total")) != (1e-4, 100, steps):
        bad.append(f"built lr schedule {sch} != lr 1e-4 / warmup 100 / total {steps}")
    if (lv.get("p_gt_start"), lv.get("p_gt_end")) != (0.2, 0.2):
        bad.append(f"built p_gt {lv.get('p_gt_start')} -> {lv.get('p_gt_end')}, expected 0.2 -> 0.2")
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
        bad.append(f"window start {(i.get('window_start'), i.get('window_t_min'), i.get('ds_window_start'))} != ({exp['window_start']},) * 3")
    if i.get("n_windows_all_clips") != exp["windows"]:
        bad.append(f"n_windows_all_clips {i.get('n_windows_all_clips')} != {exp['windows']}")
    if a.b_summary:
        b = json.load(open(a.b_summary, encoding="utf-8"))
        bl, wd = b["levers"], lv.get("w") or {}
        if i["train_sha256"] != b["train_sha256"]:
            bad.append(f"TRAIN list {i['train_sha256'][:12]} != B's {b['train_sha256'][:12]}")
        for k in ("w_pen", "w_pen_self", "w_wm"):
            if wd.get(k) != bl["w"][k]:
                bad.append(f"{k} {wd.get(k)} != B's {bl['w'][k]}")
        if (wd.get("w_tac"), wd.get("w_cons")) != (1.0, 0.5):
            bad.append(f"w_tac / w_cons {wd.get('w_tac')} / {wd.get('w_cons')} != 1.0 / 0.5 (a second lever moved)")
        if i.get("n_windows_all_clips") != b.get("n_windows"):
            bad.append(f"the window set is not B's: {i.get('n_windows_all_clips')} != {b.get('n_windows')}")
        pg, pb = lv["params"], bl["params"]
        if lv["n_params_trainable"] != bl["n_params_trainable"]:
            bad.append(f"n_params_trainable {lv['n_params_trainable']} != B's {bl['n_params_trainable']}")
        for k in sorted(set(pg) | set(pb)):
            if pg.get(k, 0) != pb.get(k, 0):
                bad.append(f"parameter group {k}: {pg.get(k)} vs B's {pb.get(k)}")
    # ---- the queue's own printed lines: every lever NAMED exactly once with its registered value
    want = {"--arm": ["M2b"], "--seed": [str(seed)], "--grid": ["pool4x10"], "--steps": [str(steps)], "--lr": ["1e-4"], "--warmup": ["100"], "--p-gt-start": ["0.2"], "--p-gt-end": ["0.2"], "--ego-perturb-p": ["0"],
            "--dagger-p": [a.p if cfg["p"] else "0"]}
    if arm == "dagoacc":
        want["--w-acc"] = [a.w_acc]
    else:
        want["--o-interface"] = ["cut"]
    mine = {"--w-acc"} if arm == "dagoacc" else {"--o-interface"}
    for name, line in (("train", tl), ("gate", gl), ("pin", pin)):
        fl = flags(line)
        for k, w in want.items():
            if fl.get(k) != [[w[0]]]:
                bad.append(f"the {name} line does not carry {k} {w[0]} exactly once (it has {fl.get(k)})")
        ik = fl.get("--init-ckpt") or []
        if len(ik) != 1 or len(ik[0]) != 1 or not ik[0][0].replace("\\", "/").endswith(itail):
            bad.append(f"the {name} line --init-ckpt {ik} is not ...{itail} exactly once")
        for k in EGO_LAW:
            if k in fl:
                bad.append(f"the {name} line names {k}: --ego-perturb-p is 0 on every DAgger arm")
        for k in DAG_ONLY:
            if not cfg["p"] and k in fl:
                bad.append(f"the {name} line names {k} on a p = 0 arm: the trainer's gate refuses a DAgger flag set while p is 0")
            if cfg["p"] and len(fl.get(k) or []) != 1:
                bad.append(f"the {name} line does not name {k} exactly once (it has {fl.get(k)})")
        if cfg["p"]:
            if fl.get("--dagger-bank-sha256") != [[a.bank_sha.lower()]]:
                bad.append(f"the {name} line's --dagger-bank-sha256 {fl.get('--dagger-bank-sha256')} != the pulled bank.json's {a.bank_sha[:12]}")
            if fl.get("--dagger-seed") != [[str(seed)]]:
                bad.append(f"the {name} line's --dagger-seed {fl.get('--dagger-seed')} != {seed}")
        for fx in SECOND_LEVER + (("--o-interface",) if arm == "dagoacc" else ("--w-acc",)):
            if fx in fl and fx not in mine:
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
    if not i.get("lru"):
        bad.append("info.lru is not recorded in the selfcheck")
    # ---- THE ONE-LEVER RULE on the argv text (C-O1 / C-F2 / FLAT's every other flag)
    if a.ref_argv_json:
        ref = json.load(open(a.ref_argv_json, encoding="utf-8")).get("argv")
        if not isinstance(ref, list) or not ref:
            bad.append("--ref-argv-json carries no argv list")
        elif arm == "dagoacc":
            if not a.dag_pin_line:
                bad.append("dagoacc: --dag-pin-line (box 2's DAG pin) is required for the one-lever rule")
            else:
                dpin = argv_of(open(a.dag_pin_line, encoding="utf-8").read())
                if dpin != ref:
                    bad.append("box 2's DAG pin is not DAG_s's banked argv.json argv: the comparator is not the banked run")
                mine_ = drop(pin)
                if mine_[-2:] != ["--w-acc", a.w_acc] or mine_[:-2] != drop(dpin):
                    bad.append("ONE-LEVER RULE: the dagoacc pin is not DAG's pin + `--w-acc W` at the end with only --out differing (C-O1)")
        else:
            if flags(drop(pin, FINE_TUNE | {"--o-interface"})) != flags(drop(ref, FINE_TUNE | {"--o-interface"})) or flags(drop(pin, FINE_TUNE)).get("--o-interface") != flags(drop(ref, FINE_TUNE)).get("--o-interface"):
                bad.append("FLAT's every other flag: the fine-tune argv minus the fine-tune / bank flags is not FLAT's banked argv (data flags, grid, arm, seed, --o-interface cut)")
    print(f"OAF gate: {'PASS' if not bad else bad} | {arm} s{seed} | p {p} | clips {i['n_clips']} steps {sc.get('steps_total')} windows {i.get('n_windows_all_clips')} | init {str(ic.get('sha256'))[:12]} "
          f"| probe grad diff {(sc.get('dagger_probe') or {}).get('grad_diff_all')} | params {lv['n_params_trainable']} | lru {i.get('lru')}")
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
    cfg = ARMS[a.arm]
    tok = {"kind": "launch-gate-token", "verdict": "PASS", "prereg": cfg["prereg"], "run": a.run,
           "binding": {"commit": a.commit, "argv_sha256": argv_sha256(t_argv), "data_sha256": data},
           "binding_note": "argv_sha256 = launch_gate.argv_sha256 of the TRAINING argv as PINNED BY BOX 2 at gate time (the file traindag executes), which equals the queue's planned training line (asserted); argv.json will record it",
           "gate": {"kind": "v8f_mini_train.py --selfcheck-only (in-process: G-DVB FLAG_CHECKS + bad_dagger + G-LIVE + dagger_probe" + (" + acc_probe)" if a.arm == "dagoacc" else " + o_interface_probe)"),
                    "argv_sha256": argv_sha256(g_argv), "argv_sha16": sc.get("argv_sha"), "selfcheck_sha256": hashlib.sha256(open(a.selfcheck, "rb").read()).hexdigest()},
           "queue_sha256": a.queue_sha, "key_files_sha256": json.loads(a.key_sha_json), "excluded_sha256": sc["info"]["excluded_sha256"], "init_ckpt_sha256": sc["levers"]["init_ckpt"]["sha256"],
           "bank_sha256": a.bank_sha.lower() if cfg["p"] else None, "dagger_p": float(a.p) if cfg["p"] else 0.0, "w_acc": sc["levers"].get("w_acc"), "o_interface": sc["levers"].get("o_interface", "full"),
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
    ap.add_argument("--arm", choices=tuple(ARMS), required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--p", default="")
    ap.add_argument("--bank-sha", default="")
    ap.add_argument("--bank-json", default="")
    ap.add_argument("--banked-init-sha", required=True)
    ap.add_argument("--w-acc", default="")
    ap.add_argument("--dag-pin-line", default="")
    ap.add_argument("--ref-argv-json", default="")
    ap.add_argument("--b-summary", default="")
    ap.add_argument("--expect", default="")
    ap.add_argument("--commit", default="")
    ap.add_argument("--queue-sha", default="")
    ap.add_argument("--key-sha-json", default="{}")
    ap.add_argument("--run", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if ARMS[a.arm]["p"] and not (a.p and a.bank_sha and a.bank_json):
        print(f"REFUSED: {a.arm} needs --p, --bank-sha and --bank-json")
        return 2
    if not ARMS[a.arm]["p"] and (a.p or a.bank_sha):
        print(f"REFUSED: {a.arm} takes no --p and no --bank-sha (a lever declared and inert)")
        return 2
    if a.arm == "dagoacc" and not a.w_acc:
        print("REFUSED: dagoacc needs --w-acc")
        return 2
    if a.arm != "dagoacc" and a.w_acc:
        print(f"REFUSED: {a.arm} takes no --w-acc")
        return 2
    bad = check(a)
    if bad:
        return 1
    return write_token(a) if a.mode == "token" else 0


if __name__ == "__main__":
    sys.exit(main())
