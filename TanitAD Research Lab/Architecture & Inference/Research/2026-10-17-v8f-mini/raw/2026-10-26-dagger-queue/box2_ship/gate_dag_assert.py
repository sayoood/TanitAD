#!/usr/bin/env python3
"""PREREG_V8F_DAGGER on box 2: the assertions the chain makes on a PULLED launch-gate selfcheck.json, and the launch-gate token that exists only after they pass. The DAgger counterpart of chain_v8f_orc.sh's
`gate ORC assertions` block (kept in a file so a test can run it on a REAL selfcheck.json, which the heredoc form never allowed).

    gate_dag_assert.py check  --selfcheck SC --gate-line F --train-line F --arm dag|dag0 --seed S [--p P] --bank-sha SHA --bank-json bank.json --banked-init-sha SHA
                              --pin-line F [--b-summary B.json] [--expect expect.json]
    gate_dag_assert.py token  (the same arguments) --commit C40 --queue-sha SHA --key-sha-json JSON --orc-read-sha SHA --out token.json

``--gate-line`` / ``--train-line`` = the lines the QUEUE PRINTS under V8F_DRYRUN (planned, with box 2's paths); ``--pin-line`` = the TRAINING line box 2 itself pinned (pulled from %RUNSM%\\dagpin\\ after the gate):
the planned train line, the pinned line and (after training) argv.json must all be the SAME argv. ``--expect`` carries the numbers that belong to the REAL corpus (clips, windows, steps, holdout); a test
that runs this on a synthetic world passes the synthetic ones. Exit 0 = every assertion holds; 1 = the list of failures is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys

REAL = {"steps": 3000, "n_clips": 4512, "n_excluded": 60, "holdout_sha": "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9", "windows": 306566, "hist": 4, "hmax": 16, "window_start": 3,
        "check_b_levers": True, "bank_dir_tail": "/dagger_bank/s{seed}/bank.json", "init_tail": "/M2b_s{seed}/ckpt_last.pt"}
SECOND_LEVER = ("--hist", "--hmax", "--window-start", "--w-tac", "--w-cons", "--w-traj", "--w-pen", "--w-pen-self", "--w-wm", "--w-acc", "--code-mix", "--o-interface", "--data-frac", "--max-clips", "--cons-absent", "--wm", "--navonly",
                "--no-consistency", "--detach-upper", "--no-resume")
EGO_LAW = ("--ego-perturb-vscale", "--ego-perturb-vmax", "--ego-perturb-adrop", "--ego-perturb-yawrate-std", "--ego-perturb-seed")
DAG_ONLY = ("--dagger-bank", "--dagger-bank-sha256", "--dagger-seed")


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


def check(a) -> list:
    sc = json.load(open(a.selfcheck, encoding="utf-8"))
    exp = {**REAL, **(json.load(open(a.expect, encoding="utf-8")) if a.expect else {})}
    seed, arm = int(a.seed), a.arm
    p = float(a.p) if arm == "dag" else 0.0
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
    ep = sc.get("ego_perturb_probe")
    if not isinstance(ep, dict) or ep.get("pass") is not True or ep.get("ego_perturb") != "off":
        bad.append(f"ego_perturb_probe {ep}: DAgger arms never combine with --ego-perturb-p (it must read 'off')")
    if (lv.get("ego_perturb") or {}).get("ego_perturb_p") != 0.0:
        bad.append(f"built ego_perturb_p {(lv.get('ego_perturb') or {}).get('ego_perturb_p')!r} != 0.0")
    # ---- THE LEVER, from the BUILT swap
    pr = sc.get("dagger_probe")
    if not isinstance(pr, dict) or pr.get("pass") is not True:
        bad.append(f"dagger_probe {pr}")
    elif arm == "dag":
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
        bad.append(f"DAG-0 (p = 0): the dagger probe should read 'off' (nothing is applied), it read {pr.get('dagger')!r}")
    if lv.get("dagger_p") != p:
        bad.append(f"built dagger_p {lv.get('dagger_p')!r} != {p!r}")
    dg = lv.get("dagger") or {}
    if arm == "dag":
        bsha = a.bank_sha.lower()
        if lv.get("dagger_bank_sha256") != bsha or lv.get("dagger_bank_sha256_declared") != bsha:
            bad.append(f"built / declared bank sha256 {str(lv.get('dagger_bank_sha256'))[:12]} / {str(lv.get('dagger_bank_sha256_declared'))[:12]} != the pulled bank.json's {bsha[:12]}")
        if dg.get("dagger_seed") != seed:
            bad.append(f"built dagger_seed {dg.get('dagger_seed')!r} != {seed}")
        if dg.get("kin_mode") != "yaw":
            bad.append(f"built DAgger kin_mode {dg.get('kin_mode')!r} != 'yaw'")
        bp = str(dg.get("dagger_bank", "")).replace("\\", "/")
        if not bp.endswith(exp["bank_dir_tail"].format(seed=seed)):
            bad.append(f"built bank path {bp!r} does not end with {exp['bank_dir_tail'].format(seed=seed)}")
        d = (i.get("dagger") or {})
        if d.get("bank_sha256") != bsha:
            bad.append(f"info.dagger.bank_sha256 {str(d.get('bank_sha256'))[:12]} != {bsha[:12]}")
        kb2 = d.get("kb2") or {}
        if not (kb2.get("ckpt_sha256") == kb2.get("init_ckpt_sha256") == a.banked_init_sha):
            bad.append(f"K-B2: bank's checkpoint {str(kb2.get('ckpt_sha256'))[:12]} / init {str(kb2.get('init_ckpt_sha256'))[:12]} != the banked M2b_s{seed} {a.banked_init_sha[:12]}")
        cov = d.get("coverage") or {}
        if not (cov.get("n_windows") and cov["n_eligible"] / cov["n_windows"] >= 0.5 and cov.get("blocked") is False):
            bad.append(f"COVERAGE in the gate: {cov}")
        kb3 = d.get("kb3") or {}
        if kb3.get("overlap_with_eval") != 0:
            bad.append(f"K-B3: overlap with eval-141 {kb3.get('overlap_with_eval')!r} != 0")
        bj = json.load(open(a.bank_json, encoding="utf-8"))
        if kb3.get("window_list_sha256") != bj["common"]["window_list_sha256"]:
            bad.append("K-B3: the trainer's TRAIN window-list sha256 is not the one the bank recorded")
        if exp["windows"] is not None and cov.get("n_windows") != exp["windows"]:
            bad.append(f"the gate's COVERAGE denominator {cov.get('n_windows')} != the TRAIN window count {exp['windows']}")
    else:
        if lv.get("dagger_bank_sha256") is not None or lv.get("dagger_bank_sha256_declared") is not None or dg.get("dagger_bank") is not None:
            bad.append("DAG-0 built a bank or declared a bank hash: a lever declared and inert")
        if "dagger" in i:
            bad.append("DAG-0's info carries a DAgger control record (a bank was read at p = 0)")
    # ---- THE START STATE: the BANKED M2b checkpoint of this seed, hashed and read back
    ic = lv.get("init_ckpt") or {}
    if ic.get("sha256") != a.banked_init_sha:
        bad.append(f"the fine-tune does not start from the banked M2b_s{seed} checkpoint: init sha256 {str(ic.get('sha256'))[:16]} != {a.banked_init_sha[:16]}")
    if i.get("init_ckpt") != ic:
        bad.append("info.init_ckpt and levers.init_ckpt disagree")
    if ic.get("applied") is not True or ic.get("n_mismatch_after_load") != 0 or not ic.get("n_tensors"):
        bad.append(f"init_ckpt not applied / read back: applied {ic.get('applied')} mismatches {ic.get('n_mismatch_after_load')} tensors {ic.get('n_tensors')}")
    itail = exp["init_tail"].format(seed=seed)
    if not str(ic.get("path", "")).replace("\\", "/").endswith(itail):
        bad.append(f"init_ckpt path {ic.get('path')!r} is not ...{itail}")
    # ---- THE FINE-TUNE'S SCHEDULE (ORC Amendment 1, unchanged by PREREG_V8F_DAGGER section 4)
    steps = exp["steps"]
    if sc.get("steps_total") != steps:
        bad.append(f"steps {sc.get('steps_total')} != the PREREG's {steps}")
    sch = lv.get("lr_schedule") or {}
    if (sch.get("lr"), sch.get("warmup"), sch.get("total")) != (1e-4, 100, steps):
        bad.append(f"built lr schedule {sch} != lr 1e-4 / warmup 100 / total {steps}")
    if (lv.get("p_gt_start"), lv.get("p_gt_end")) != (0.2, 0.2):
        bad.append(f"built p_gt {lv.get('p_gt_start')} -> {lv.get('p_gt_end')}, expected 0.2 -> 0.2")
    # ---- B's DATA AND MODEL are the fine-tune's
    if exp["n_clips"] is not None and (i["n_clips"], i["n_excluded"], i["n_excluded_declared"], i["excluded_in_train"]) != (exp["n_clips"], exp["n_excluded"], exp["n_excluded"], 0):
        bad.append("clips / holdout counts")
    if exp["holdout_sha"] is not None and i["excluded_sha256"] != exp["holdout_sha"]:
        bad.append("the excluded set is not the RA-1 holdout")
    if i["kin_mode"] != "yaw":
        bad.append("kin_mode")
    if (lv["wm"], lv["navonly"], lv["consistency"], lv.get("cons_absent")) != ("b", False, True, "mask") and exp["check_b_levers"]:
        bad.append(f"levers wm / navonly / consistency / cons_absent {(lv['wm'], lv['navonly'], lv['consistency'], lv.get('cons_absent'))}")
    if lv.get("o_interface", "full") != "full":
        bad.append(f"built o_interface {lv.get('o_interface')!r} != 'full' (B's)")
    if lv.get("code_mix", "whole") != "whole":
        bad.append(f"built code_mix {lv.get('code_mix')!r} != 'whole' (B's)")
    if lv.get("w_acc", 0.0) != 0.0:
        bad.append(f"built w_acc {lv.get('w_acc')!r} != 0.0 (B's)")
    if exp["check_b_levers"]:
        if (lv.get("hist"), lv.get("hmax"), lv.get("window_start")) != (exp["hist"], exp["hmax"], exp["window_start"]):
            bad.append(f"built hist / hmax / window_start {(lv.get('hist'), lv.get('hmax'), lv.get('window_start'))} != {(exp['hist'], exp['hmax'], exp['window_start'])} (B's)")
        if (i.get("window_start"), i.get("window_t_min"), i.get("ds_window_start")) != (exp["window_start"],) * 3:
            bad.append(f"window start (asked / first t / dataset clamp origin) {(i.get('window_start'), i.get('window_t_min'), i.get('ds_window_start'))} != ({exp['window_start']},) * 3")
    if exp["windows"] is not None and i.get("n_windows_all_clips") != exp["windows"]:
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
        if i.get("n_windows_all_clips") != b.get("n_windows"):
            bad.append(f"the window set is not B's: n_windows_all_clips {i.get('n_windows_all_clips')} != B's {b.get('n_windows')}")
        pg, pb = lv["params"], bl["params"]
        if lv["n_params_trainable"] != bl["n_params_trainable"]:
            bad.append(f"n_params_trainable {lv['n_params_trainable']} != B's {bl['n_params_trainable']}")
        for k in sorted(set(pg) | set(pb)):
            if pg.get(k, 0) != pb.get(k, 0):
                bad.append(f"parameter group {k}: {pg.get(k)} vs B's {pb.get(k)}")
    # ---- THE QUEUE'S OWN PRINTED LINES: every lever NAMED exactly once with its registered value (the flags' defaults disable the lever), nothing else moved
    want = {"--arm": ["M2b"], "--seed": [str(seed)], "--grid": ["pool4x10"], "--steps": [str(steps)], "--lr": ["1e-4"], "--warmup": ["100"], "--p-gt-start": ["0.2"], "--p-gt-end": ["0.2"], "--ego-perturb-p": ["0"],
            "--dagger-p": [a.p if arm == "dag" else "0"]}
    for name, line in (("train", tl), ("gate", gl), ("pin", pin)):
        fl = flags(line)
        for k, w in want.items():
            if fl.get(k) != [[w[0]]]:
                bad.append(f"the {name} line does not carry {k} {w[0]} exactly once (it has {fl.get(k)})")
        ik = fl.get("--init-ckpt") or []
        if len(ik) != 1 or len(ik[0]) != 1 or not ik[0][0].replace("\\", "/").endswith(exp["init_tail"].format(seed=seed)):
            bad.append(f"the {name} line --init-ckpt {ik} is not ...{exp['init_tail'].format(seed=seed)} exactly once")
        for k in EGO_LAW:
            if k in fl:
                bad.append(f"the {name} line names {k}: --ego-perturb-p is 0 on both DAgger arms (the trainer's gate refuses a law flag at p = 0)")
        for k in DAG_ONLY:
            if arm == "dag0" and k in fl:
                bad.append(f"the {name} line names {k} on DAG-0: the trainer's gate refuses a DAgger flag set while p is 0 (PREREG_V8F_DAGGER section 4: DAG-0 names only --dagger-p 0)")
            if arm == "dag" and len(fl.get(k) or []) != 1:
                bad.append(f"the {name} line does not name {k} exactly once on DAG (it has {fl.get(k)})")
        if arm == "dag":
            f = flags(line)
            if f.get("--dagger-bank-sha256") != [[a.bank_sha.lower()]]:
                bad.append(f"the {name} line's --dagger-bank-sha256 {f.get('--dagger-bank-sha256')} != the pulled bank.json's {a.bank_sha[:12]}")
            if f.get("--dagger-seed") != [[str(seed)]]:
                bad.append(f"the {name} line's --dagger-seed {f.get('--dagger-seed')} != {seed}")
        for fx in SECOND_LEVER:
            if fx in fl:
                bad.append(f"the {name} line sets {fx}: a second lever beside the fine-tune")
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
        bad.append("info.lru is not recorded in the selfcheck (the pipeline sizing must be on the record)")
    print("DAG gate:", "PASS" if not bad else bad, "|", arm, "s%d" % seed, "| p", p, "| clips", i["n_clips"], "steps", sc.get("steps_total"), "windows", i.get("n_windows_all_clips"), "| init", str(ic.get("sha256"))[:12],
          "| probe grad diff", (pr or {}).get("grad_diff_all"), "| params", lv["n_params_trainable"], "| lru", i.get("lru"))
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
    tok = {"kind": "launch-gate-token", "verdict": "PASS", "prereg": "PREREG_V8F_DAGGER", "run": a.run,
           "binding": {"commit": a.commit, "argv_sha256": argv_sha256(t_argv), "data_sha256": data},
           "binding_note": "argv_sha256 = launch_gate.argv_sha256 of the TRAINING argv as PINNED BY BOX 2 at gate time (the file traindag executes), which equals the queue's planned training line (asserted); argv.json will record it",
           "gate": {"kind": "v8f_mini_train.py --selfcheck-only (in-process: G-DVB FLAG_CHECKS + bad_dagger + G-LIVE + dagger_probe)", "argv_sha256": argv_sha256(g_argv), "argv_sha16": sc.get("argv_sha"),
                    "selfcheck_sha256": hashlib.sha256(open(a.selfcheck, "rb").read()).hexdigest()},
           "queue_sha256": a.queue_sha, "key_files_sha256": json.loads(a.key_sha_json), "excluded_sha256": sc["info"]["excluded_sha256"], "init_ckpt_sha256": sc["levers"]["init_ckpt"]["sha256"],
           "bank_sha256": a.bank_sha.lower() if a.arm == "dag" else None, "orc_read_sha256": a.orc_read_sha, "dagger_p": float(a.p) if a.arm == "dag" else 0.0,
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
    ap.add_argument("--arm", choices=("dag", "dag0"), required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--p", default="")
    ap.add_argument("--bank-sha", default="")
    ap.add_argument("--bank-json", default="")
    ap.add_argument("--banked-init-sha", required=True)
    ap.add_argument("--b-summary", default="")
    ap.add_argument("--expect", default="")
    ap.add_argument("--commit", default="")
    ap.add_argument("--queue-sha", default="")
    ap.add_argument("--key-sha-json", default="{}")
    ap.add_argument("--orc-read-sha", default="")
    ap.add_argument("--run", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.arm == "dag" and not (a.p and a.bank_sha and a.bank_json):
        print("REFUSED: dag needs --p, --bank-sha and --bank-json")
        return 2
    if a.arm == "dag0" and (a.p or a.bank_sha):
        print("REFUSED: dag0 takes no --p and no --bank-sha (a lever declared and inert)")
        return 2
    bad = check(a)
    if bad:
        return 1
    return write_token(a) if a.mode == "token" else 0


if __name__ == "__main__":
    sys.exit(main())
