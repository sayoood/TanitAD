#!/usr/bin/env python3
"""PREREG_V8F_DAG_ORC on box 2 (derived from gate_dag2_assert.py and the ORC chain's `gate ORC assertions` block; arms dagorc | dagorcrep | dagorc01): the assertions the chain makes on a PULLED launch-gate selfcheck.json, and the
launch-gate token that exists only after they pass.

    gate_dagorc_assert.py check --selfcheck SC --gate-line F --train-line F --pin-line F --arm dagorc|dagorcrep|dagorc01 --seed S --banked-init-sha SHA [--b-summary DAG2-0 summary.json]
                                [--ref-argv-json DAG2-0 argv.json] [--expect expect.json]
    gate_dagorc_assert.py token (the same arguments) --commit C40 --queue-sha SHA --key-sha-json JSON --run NAME --out token.json

The three arms (PREREG_V8F_DAG_ORC section 1) and what each must be, against DAG2-0's BANKED argv (``--ref-argv-json``, the one-lever reference; its sha256 is pinned by the chain):
  dagorc     (A)    ref with ``--ego-perturb-p 0`` replaced by ``--ego-perturb-p 0.3`` + the five law flags right after it (vscale 0.5 1.0, vmax 8.0, adrop 1.5, yawrate-std 2.0, seed = SEED); only --out differs otherwise
  dagorc01   (B)    the same with p 0.1
  dagorcrep  (REP)  ref with ``--seed`` = 10 + SEED; only --out and --seed differ; no perturbation
``--gate-line`` / ``--train-line`` = the lines the QUEUE PRINTS under V8F_DRYRUN (planned, with box 2's paths); ``--pin-line`` = the TRAINING line box 2 itself pinned (pulled from %RUNSM%\\dagpin\\ after the gate): the planned
train line, the pinned line and (after training) argv.json must all be the SAME argv. ``--expect`` carries the numbers that belong to the REAL corpus (clips, windows, steps, holdout, the init path tail); a test that runs
this on a synthetic world passes the synthetic ones. Without ``--ref-argv-json`` the ONE-LEVER rule is skipped (printed): only a synthetic world has no banked reference. Exit 0 = every assertion holds; 1 = the list of failures is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys

REAL = {"steps": 3000, "warmup": 100, "n_clips": 4512, "n_excluded": 60, "holdout_sha": "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9", "windows": 306566, "hist": 4, "hmax": 16, "window_start": 3,
        "check_b_levers": True, "init_tail": "/M2b_s{seed}_dag/ckpt_last.pt"}
SECOND_LEVER = ("--hist", "--hmax", "--window-start", "--w-tac", "--w-cons", "--w-traj", "--w-pen", "--w-pen-self", "--w-wm", "--w-acc", "--code-mix", "--o-interface", "--data-frac", "--max-clips", "--cons-absent", "--wm", "--navonly",
                "--no-consistency", "--detach-upper", "--no-resume")
EGO_LAW = ("--ego-perturb-vscale", "--ego-perturb-vmax", "--ego-perturb-adrop", "--ego-perturb-yawrate-std", "--ego-perturb-seed")
DAG_ONLY = ("--dagger-bank", "--dagger-bank-sha256", "--dagger-seed")
ARMS = {"dagorc": 0.3, "dagorc01": 0.1, "dagorcrep": 0.0}      # arm -> the registered p (PREREG_V8F_DAG_ORC section 1)
LAW_VALUES = {"ego_perturb_vscale": [0.5, 1.0], "ego_perturb_vmax": 8.0, "ego_perturb_adrop": 1.5, "ego_perturb_yawrate_std": 2.0}      # + ego_perturb_seed == SEED
LAW_DEFAULT = {"ego_perturb_vscale": [0.5, 1.0], "ego_perturb_vmax": 8.0, "ego_perturb_adrop": 1.5, "ego_perturb_yawrate_std": 0.0, "ego_perturb_seed": 0}      # the trainer's defaults, built when no law flag is named (p = 0)
PROBE_CONTROLS = ("control_repeat_loss_diff", "control_repeat_grad_diff", "control_p0_loss_diff", "control_p0_grad_diff", "control_prior_grad_diff")


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


def expected_from_ref(ref: list, arm: str, seed: int) -> list:
    """The registered argv of ``arm`` built from DAG2-0's banked argv (``ref``, ``--out`` dropped): A / B = ``--ego-perturb-p 0`` -> ``--ego-perturb-p <p>`` + the five law flags right after it; REP = ``--seed`` -> 10 + SEED.
    Raises ValueError when the reference does not have the shape the rule needs (one ``--ego-perturb-p 0``, one ``--seed SEED``)."""
    r = drop(ref)
    if r.count("--ego-perturb-p") != 1 or r.count("--seed") != 1:
        raise ValueError("the reference argv must name --ego-perturb-p and --seed exactly once")
    kp, ks = r.index("--ego-perturb-p"), r.index("--seed")
    if r[kp + 1] != "0" or r[ks + 1] != str(seed):
        raise ValueError(f"the reference argv must carry --ego-perturb-p 0 and --seed {seed}, it has {r[kp + 1]!r} and {r[ks + 1]!r}")
    if arm == "dagorcrep":
        out = list(r)
        out[ks + 1] = str(10 + seed)
        return out
    p = {"dagorc": "0.3", "dagorc01": "0.1"}[arm]
    law = ["--ego-perturb-vscale", "0.5", "1.0", "--ego-perturb-vmax", "8.0", "--ego-perturb-adrop", "1.5", "--ego-perturb-yawrate-std", "2.0", "--ego-perturb-seed", str(seed)]
    return r[:kp + 1] + [p] + law + r[kp + 2:]


def check(a) -> list:
    sc = json.load(open(a.selfcheck, encoding="utf-8"))
    exp = {**REAL, **(json.load(open(a.expect, encoding="utf-8")) if a.expect else {})}
    seed, arm = int(a.seed), a.arm
    p = ARMS[arm]
    tseed = 10 + seed if arm == "dagorcrep" else seed      # the trainer's --seed
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
    # ---- THE LEVER (A, B): G-LIVE on a PERTURBED batch; the perturbation REACHES the loss the trainer backpropagates and every control reads EXACTLY 0.0. REP: the identity ('off')
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
    elif pr.get("ego_perturb") != "off":
        bad.append(f"REP (p = 0): the ego probe should read 'off' (nothing is applied), it read {pr.get('ego_perturb')!r}")
    ep = lv.get("ego_perturb") or {}
    want_ep = {"ego_perturb_p": p, **({**LAW_VALUES, "ego_perturb_seed": seed} if p > 0.0 else LAW_DEFAULT), "kin_mode": "yaw"}
    for k, w in want_ep.items():
        if ep.get(k) != w:
            bad.append(f"built ego_perturb {k} {ep.get(k)!r} != {w!r}")
    # ---- THE SWAP IS OFF ON ALL THREE ARMS (the trainer's gate refuses the two levers together; DAG-ORC names --dagger-p 0 and no bank)
    dp = sc.get("dagger_probe")
    if not isinstance(dp, dict) or dp.get("pass") is not True or dp.get("dagger") != "off":
        bad.append(f"dagger_probe {dp}: the DAgger swap is off on every DAG-ORC arm (it must read 'off')")
    if lv.get("dagger_p") != 0.0:
        bad.append(f"built dagger_p {lv.get('dagger_p')!r} != 0.0")
    if lv.get("dagger_bank_sha256") is not None or lv.get("dagger_bank_sha256_declared") is not None or (lv.get("dagger") or {}).get("dagger_bank") is not None:
        bad.append("a DAgger bank or bank hash was built or declared: a lever declared and inert")
    if "dagger" in i:
        bad.append("info carries a DAgger control record (a bank was read at p = 0)")
    # ---- THE START STATE: DAG_s<seed>'s checkpoint, hashed and read back (K-INIT)
    ic = lv.get("init_ckpt") or {}
    if ic.get("sha256") != a.banked_init_sha:
        bad.append(f"the fine-tune does not start from DAG_s{seed} checkpoint: init sha256 {str(ic.get('sha256'))[:16]} != {a.banked_init_sha[:16]}")
    if i.get("init_ckpt") != ic:
        bad.append("info.init_ckpt and levers.init_ckpt disagree")
    if ic.get("applied") is not True or ic.get("n_mismatch_after_load") != 0 or not ic.get("n_tensors"):
        bad.append(f"init_ckpt not applied / read back: applied {ic.get('applied')} mismatches {ic.get('n_mismatch_after_load')} tensors {ic.get('n_tensors')}")
    itail = exp["init_tail"].format(seed=seed)
    if not str(ic.get("path", "")).replace("\\", "/").endswith(itail):
        bad.append(f"init_ckpt path {ic.get('path')!r} is not ...{itail}")
    # ---- THE FINE-TUNE'S SCHEDULE (ORC Amendment 1, DAG2-0's)
    steps = exp["steps"]
    if sc.get("steps_total") != steps:
        bad.append(f"steps {sc.get('steps_total')} != the PREREG's {steps}")
    sch = lv.get("lr_schedule") or {}
    if (sch.get("lr"), sch.get("warmup"), sch.get("total")) != (1e-4, exp["warmup"], steps) and exp["check_b_levers"]:
        bad.append(f"built lr schedule {sch} != lr 1e-4 / warmup 100 / total {steps}")
    if (lv.get("p_gt_start"), lv.get("p_gt_end")) != (0.2, 0.2):
        bad.append(f"built p_gt {lv.get('p_gt_start')} -> {lv.get('p_gt_end')}, expected 0.2 -> 0.2")
    # ---- the comparator "B" is DAG2-0's summary.json: same data, same loss weights, same parameter groups
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
    want = {"--arm": ["M2b"], "--seed": [str(tseed)], "--grid": ["pool4x10"], "--steps": [str(steps)], "--lr": ["1e-4"], "--warmup": [str(exp["warmup"])], "--p-gt-start": ["0.2"], "--p-gt-end": ["0.2"],
            "--ego-perturb-p": [{0.3: "0.3", 0.1: "0.1", 0.0: "0"}[p]], "--dagger-p": ["0"]}
    law_want = {"--ego-perturb-vscale": ["0.5", "1.0"], "--ego-perturb-vmax": ["8.0"], "--ego-perturb-adrop": ["1.5"], "--ego-perturb-yawrate-std": ["2.0"], "--ego-perturb-seed": [str(seed)]}
    if p > 0.0:
        want.update(law_want)
    for name, line in (("train", tl), ("gate", gl), ("pin", pin)):
        fl = flags(line)
        for k, w in want.items():
            if fl.get(k) != [w]:
                bad.append(f"the {name} line does not carry {k} {' '.join(w)} exactly once (it has {fl.get(k)})")
        ik = fl.get("--init-ckpt") or []
        if len(ik) != 1 or len(ik[0]) != 1 or not ik[0][0].replace("\\", "/").endswith(itail):
            bad.append(f"the {name} line --init-ckpt {ik} is not ...{itail} exactly once")
        if p == 0.0:
            for k in EGO_LAW:
                if k in fl:
                    bad.append(f"the {name} line names {k} on REP: the trainer's gate refuses a law flag set while p is 0 (REP names ONLY --ego-perturb-p 0)")
        for k in DAG_ONLY:
            if k in fl:
                bad.append(f"the {name} line names {k}: DAG-ORC reads no bank (--dagger-p 0 and nothing else of the DAgger flags)")
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
    # ---- THE ONE-LEVER RULE (K-PAIR) against DAG2-0's banked argv: A / B = ref + the perturbation exactly; REP = ref with --seed only; only --out differs
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
                    bad.append(f"ONE-LEVER: the {name} line is not DAG2-0's banked argv + exactly the registered change for {arm} (first difference at token {k}: {drop(line)[k:k + 2]} vs {exp_argv[k:k + 2]}; only --out may differ)")
                    onelever = "NOT"
    if not i.get("lru"):
        bad.append("info.lru is not recorded in the selfcheck (the pipeline sizing must be on the record)")
    print("DAG-ORC gate:", "PASS" if not bad else bad, "|", arm, "s%d" % seed, "| p", p, "| trainer seed", tseed, "| clips", i["n_clips"], "steps", sc.get("steps_total"), "windows", i.get("n_windows_all_clips"),
          "| init", str(ic.get("sha256"))[:12], "| probe grad diff", (pr or {}).get("grad_diff_all"), "| params", lv["n_params_trainable"], "| lru", i.get("lru"), "| one-lever:", onelever)
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
    tok = {"kind": "launch-gate-token", "verdict": "PASS", "prereg": "PREREG_V8F_DAG_ORC", "run": a.run,
           "binding": {"commit": a.commit, "argv_sha256": argv_sha256(t_argv), "data_sha256": data},
           "binding_note": "argv_sha256 = launch_gate.argv_sha256 of the TRAINING argv as PINNED BY BOX 2 at gate time (the file traindag executes), which equals the queue's planned training line (asserted); argv.json will record it",
           "gate": {"kind": "v8f_mini_train.py --selfcheck-only (in-process: G-DVB FLAG_CHECKS + bad_ego_perturb + bad_init_ckpt + G-LIVE + ego_perturb_probe)", "argv_sha256": argv_sha256(g_argv), "argv_sha16": sc.get("argv_sha"),
                    "selfcheck_sha256": hashlib.sha256(open(a.selfcheck, "rb").read()).hexdigest()},
           "queue_sha256": a.queue_sha, "key_files_sha256": json.loads(a.key_sha_json), "excluded_sha256": sc["info"]["excluded_sha256"], "init_ckpt_sha256": sc["levers"]["init_ckpt"]["sha256"],
           "arm": a.arm, "ego_perturb_p": ARMS[a.arm], "trainer_seed": 10 + int(a.seed) if a.arm == "dagorcrep" else int(a.seed),
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
        print("REFUSED: --seed must be 0 or 1 (the seed of DAG_s, the init)")
        return 2
    bad = check(a)
    if bad:
        return 1
    return write_token(a) if a.mode == "token" else 0


if __name__ == "__main__":
    sys.exit(main())
