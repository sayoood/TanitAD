"""STOPPERC dry-run STAND-IN for stack/scripts/v8f_mini_train.py (dev box only; shipped through a throw-away git repo by the dry run). It writes what the trainer at the code of record (3e264bbf) is documented to write:
selfcheck.json with the keys stopbal_check.gate_problems AND stopperc_check.a_problems read (levers.stop_aux / w_stop_aux / params.STOPAUX / stop_vision_only_source, stop_aux_extrinsics, stop_aux_probe.* incl. probe_n_pos and the
step0 terms the W script reads, live.terms incl. the three stop terms, info.init_ckpt.fresh_keys), train_log.jsonl with cumulative `sec`, summary.json and a torch checkpoint, from its OWN reading of the argv, steered by the json file
named in PLANT_CONTROL. CPU only. torch is imported only where a checkpoint is written.
Needle strings the ship step requires, present here so the stand-in passes it: "--goal-pos-weight-stop" "--lon-class-weight-stop" goal_pos_weight_stop lon_class_weight_stop "--w-acc" "--dagger-p" def acc_probe def dagger_probe"""
import hashlib
import json
import os
import sys
import time

import numpy as np

PARAMS = {"adapter": 275712, "cond": 25088, "S": 1321737, "T": 1463868, "O": 2120216, "WM": 2007874}
STOPAUX = 132098
STOP_KEYS = sorted(f"{h}.{i}.{p}" for h in ("stop_head", "stop_pres_head") for i in (0, 2) for p in ("weight", "bias"))
GOAL_TERM, STOP_TERM = 2.3456, 0.6789                       # step-0 unweighted terms: W = goal / stop_aux = 3.4550 (4 decimals)
TABLE_B2 = "D:\\tanitad\\stopperc\\data\\stopaux_extr_join4572.json"
MOVES = {"S": ("read_s", "head_s"), "T": ("read_t", "trunk_t", "h_goal"), "O": ("read_o", "head_o")}


def parse(argv):
    F, cur = {}, None
    for t in argv:
        if t.startswith("--"):
            cur = t[2:]
            F[cur] = []
        elif cur:
            F[cur].append(t)
    return F


def bank_sha_of(k):
    return json.load(open(os.environ["STOPBAL_FAKE_BANK_SHA_FILE"]))[str(k)]


def model_state(seed):
    """numpy tensors keyed like the real model's groups (the init checkpoint of the fake box)"""
    rng = np.random.RandomState(seed)
    keys = ["read_s.w", "head_s.w", "adapter.0.w", "pos", "time", "e_ego.w", "e_nav.w", "e_lim.w", "e_age.w", "e_ck.w", "e_code.w", "read_t.w", "trunk_t.w", "h_goal.w", "head_o.w", "read_o.w"]
    return {k: rng.randn(4).astype(np.float32) for k in keys}


def trained_state(init, seed, mode=None, stop=False):
    """what a correct run leaves: the init with every group moved (deterministic in init and seed); `stop` adds the 8 fresh head tensors"""
    st = model_state(init)
    for g, pre in MOVES.items():
        if mode == "no_move" or (mode == "S_frozen" and g == "S"):
            continue
        for k in st:
            if k.startswith(pre):
                st[k] = st[k] + 1.0 + 0.01 * seed
    if stop and mode != "head_missing":
        rng = np.random.RandomState(1000 + seed)
        for k in STOP_KEYS:
            st[k] = rng.randn(4).astype(np.float32)
    return st


def to_torch(st):
    import torch
    return {k: torch.from_numpy(np.array(v)) for k, v in st.items()}


def table_sha(path):
    h = hashlib.sha256()
    root = os.environ["STOPBAL_ROOT"].replace("\\", "/")
    p = path.replace("\\", "/")
    if p.lower().startswith("d:/tanitad"):
        p = root + p[len("d:/tanitad"):]
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def good_selfcheck(argv, mode="ok", n_pos=None, goal=None):
    """the selfcheck.json a correct build writes for this argv; `mode` plants one defect (the unit tests mutate the dict instead)"""
    F = parse(argv)
    one = lambda k, d=None: F[k][0] if F.get(k) else d                           # noqa: E731
    init = int(one("init-ckpt").replace("\\", "/").split("M2b_s")[1][0])
    on = one("stop-aux", "off") == "on"
    wsa = float(one("w-stop-aux", 0.0))
    wacc = float(one("w-acc", 0.0))
    steps = int(one("steps", 0)) or 4791 * int(one("steps-mult", 1))
    sha16 = hashlib.sha256(" ".join(argv).encode()).hexdigest()[:16]
    ish = json.load(open(os.environ["STOPBAL_FAKE_INIT_SHA_FILE"]))[str(init)]
    bsha = bank_sha_of(init)
    params = dict(PARAMS)
    ntrain = 7214495
    if on:
        params["STOPAUX"] = STOPAUX
        ntrain += STOPAUX
    lv = {"kin_mode": "yaw", "params": params, "n_params_trainable": ntrain, "w_acc": wacc, "goal_pos_weight_stop": 1.0, "lon_class_weight_stop": 1.0, "p_gt_start": 0.2, "p_gt_end": 0.2,
          "recover": {"recover_p": 0.0}, "dagger_p": float(one("dagger-p", 0.0)), "dagger_bank_sha256": bsha, "dagger_bank_sha256_declared": one("dagger-bank-sha256"),
          "dagger": {"dagger_p": float(one("dagger-p", 0.0)), "dagger_seed": int(one("dagger-seed", 0)), "dagger_bank": one("dagger-bank") + "\\bank.json", "dagger_bank_sha256": bsha},
          "ego_perturb": {"ego_perturb_p": 0.0}, "ego_perturb_with_dagger": False, "launch_intent": {"launch_intent_w": 0.0}}
    terms = {"ckpt": 1.0, "goal": 1.0, "lat": 1.0, "lon": 1.0, "cons": 1.0, "ade": 2.0, "fde": 3.0, "acc": 0.1, "total": 5.0, "wm": 1.0}
    gg = {"adapter": 9.0, "S": 3.0, "T": 1.5, "O": 2.0, "head_s": 1.0, "head_o": 2.0, "wm": 1.0}
    ic = {"path": one("init-ckpt"), "sha256": ish, "applied": True, "n_mismatch_after_load": 0, "n_tensors": 40, "source_arm": "M2b", "source_seed": init}
    sc = {"declared_vs_built": "PASS", "argv_sha": sha16, "steps_total": steps, "S_steps_per_epoch_100pct": 4791,
          "info": {"train_sha256": "37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1", "excluded_sha256": "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9", "n_clips": 4512, "n_excluded": 60,
                   "batch": 64, "md5_overridden": False, "excluded_in_train": 0, "init_ckpt": ic, "dagger": {"bank_sha256": bsha, "coverage": {"blocked": False, "fraction": 0.75}}},
          "levers": lv, "live": {"pass": True, "missing_terms": [], "dead_params": [], "unexpected_dead": [], "terms": terms, "control_prior_grad_sum": 0.0, "head_o_grad_step0": 0.7, "groups_grad": gg},
          "acc_probe": {"w_acc_declared": wacc, "w_probe": wacc, "acc": 0.01, "contribution": wacc * 0.01, "w_x_acc": wacc * 0.01, "head_o_grad_diff": 4.0, "read_o_grad_diff": 8.0,
                        "control_repeat_total_diff": 0.0, "control_repeat_grad_diff": 0.0, "control_prior_grad_diff": 0.0, "pass": True},
          "dagger_probe": {"p_declared": float(one("dagger-p", 0.0)), "n_rows": 64, "n_eligible": 49, "n_selected_at_p1": 49, "control_repeat_loss_diff": 0.0, "control_repeat_grad_diff": 0.0,
                           "control_p0_loss_diff": 0.0, "control_p0_grad_diff": 0.0, "control_prior_grad_diff": 0.0, "pass": True}}
    goal_term = GOAL_TERM if goal is None else float(goal)
    if on:
        ic["fresh_keys"], ic["declared_new_keys"] = list(STOP_KEYS), list(STOP_KEYS)
        lv["stop_aux"], lv["w_stop_aux"], lv["stop_vision_only_source"] = "on", wsa, True
        terms.update({"stop_aux": STOP_TERM, "stop_cell": 0.4, "stop_pres": STOP_TERM - 0.4})
        sc["stop_aux_extrinsics"] = {"path": one("stop-aux-extr"), "sha256": table_sha(one("stop-aux-extr")), "coverage_train": 1.0, "n_train_clips": 4512, "n_train_clips_with_extrinsics": 4512, "n_join_clips": 4572,
                                     "n_join_clips_with_extrinsics": 4572}
        npos = 3 if n_pos is None else int(n_pos)
        sc["stop_aux_probe"] = {"w_stop_aux_declared": wsa, "vision_only": {"source_problems": [], "keys_read": []}, "probe_n_has": 40, "probe_n_pos": npos, "step0_total": 9.0, "step0_stop_aux_term": STOP_TERM,
                                "step0_stop_cell": 0.4, "step0_stop_pres": STOP_TERM - 0.4, "step0_goal_term": goal_term, "w_stop_aux_for_goal_equal": goal_term / STOP_TERM, "term": STOP_TERM, "total_w": 9.0 + wsa * STOP_TERM,
                                "total_0": 9.0, "contribution": wsa * STOP_TERM, "w_x_term": wsa * STOP_TERM, "head_grad_diff": 0.5, "pres_head_grad_diff": 0.2, "adapter_grad_diff": 0.3, "head_grad_at_w0_zero_control": 0.0,
                                "control_repeat_total_diff": 0.0, "control_repeat_grad_diff": 0.0, "control_allfalse_term": 0.0, "control_allfalse_total_diff": 0.0, "control_allfalse_head_grad_diff": 0.0,
                                "control_prior_grad_diff": 0.0}
    if mode == "bad_params":
        lv["params"]["S"] = 0
    elif mode == "dead_goal_head":
        sc["live"]["dead_params"] = ["h_goal.0.weight"]
        sc["live"]["unexpected_dead"] = ["h_goal.0.weight"]
    elif mode == "bad_steps":
        sc["steps_total"] = 12345
    elif mode == "bad_bank":
        lv["dagger_bank_sha256"] = "0" * 64
    elif mode == "w_not_built" and on:                                           # the declared W never reached the built model
        lv["w_stop_aux"] = 1.0
    elif mode == "stop_off_built" and on:                                        # the flag never reached the built model
        lv["stop_aux"] = "off"
    elif mode == "no_head_params" and on:
        lv["params"]["STOPAUX"] = 0
    elif mode == "dead_stop_head" and on:
        sc["stop_aux_probe"]["head_grad_diff"] = 0.0
    elif mode == "bad_table_sha" and on:
        sc["stop_aux_extrinsics"]["sha256"] = "0" * 64
    elif mode == "null_has_stop" and not on:                                     # an OFF arm whose build carries a stop key
        sc["levers"]["stop_aux"] = "off"
        sc["stop_aux_probe"] = {"stop_aux": "off"}
    return sc


def main():
    argv = sys.argv[1:]
    F = parse(argv)
    one = lambda k, d=None: F[k][0] if F.get(k) else d                           # noqa: E731
    ctl = json.load(open(os.environ["PLANT_CONTROL"])) if os.environ.get("PLANT_CONTROL") and os.path.exists(os.environ["PLANT_CONTROL"]) else {}
    out = one("out").replace("\\", "/")
    root = os.environ["STOPBAL_ROOT"].replace("\\", "/")
    if out.lower().startswith("d:/tanitad"):
        out = root + out[len("d:/tanitad"):]
    os.makedirs(out, exist_ok=True)
    base = os.path.basename(out)
    if "selfcheck-only" in F:
        if (ctl.get("selfcheck_crash") or {}).get(base):                         # the real trainer raises on a probe it cannot read (probe_n_pos 0) and exits non-zero without a selfcheck.json
            print("stop-aux probe: cannot see the cell term (probe_n_pos 0)", file=sys.stderr)
            return 1
        mode = (ctl.get("modes") or {}).get(base, ctl.get("mode", "ok"))        # per-tag defects: the control file maps a gate out dir name to a mode
        n_pos = (ctl.get("n_pos") or {}).get(base)
        json.dump(good_selfcheck(argv, mode, n_pos, (ctl.get("goal") or {}).get(base)), open(os.path.join(out, "selfcheck.json"), "w"), indent=1)
        return 0
    tm = (ctl.get("train_modes") or {}).get(base, ctl.get("train_mode"))
    if tm == "crash":
        return 1
    sc = good_selfcheck(argv)
    on = one("stop-aux", "off") == "on"
    steps = sc["steps_total"]
    sps, startup = float(ctl.get("sps", 1.0)), float(ctl.get("startup_s", 30.0))
    every = 50 if steps <= 200 else 500
    with open(os.path.join(out, "train_log.jsonl"), "w") as f:
        for s in sorted(set(range(every, steps + 1, every)) | {steps}):
            f.write(json.dumps({"step": s, "sec": startup + sps * s}) + "\n")
    init = int(one("init-ckpt").replace("\\", "/").split("M2b_s")[1][0])
    seed = int(one("seed", 0))
    st = trained_state(init, seed, tm, on)
    cfg = {"detach_upper": False, "w_traj": 1.0, "w_acc": float(one("w-acc", 0.0)), "goal_pos_weight_stop": 1.0, "lon_class_weight_stop": 1.0}
    if on:
        cfg.update({"stop_aux": "on", "w_stop_aux": float(one("w-stop-aux", 0.0))})
        if tm == "cfg_stop_lost":                                                # the checkpoint's config does not carry the declared W
            cfg["w_stop_aux"] = 0.0
    if not on and tm == "null_cfg_stop":                                          # an OFF arm whose checkpoint config carries a stop-aux entry
        cfg["stop_aux"] = "off"
    dg = {"dagger_p": float(one("dagger-p", 0.0)), "dagger_seed": int(one("dagger-seed", 0)), "dagger_bank_sha256": bank_sha_of(init)}
    ck_seed = seed + 1 if tm == "wrong_seed" else seed
    import torch
    torch.save({"model": to_torch(st), "cfg": cfg, "step": steps, "seed": ck_seed, "arm": one("arm"), "argv_sha": sc["argv_sha"], "dagger": dg, "init_ckpt": {"sha256": sc["info"]["init_ckpt"]["sha256"]}},
               os.path.join(out, "ckpt_last.pt"))
    json.dump({"done": True, "steps": steps, "arm": one("arm"), "seed": seed, "argv_sha": sc["argv_sha"]}, open(os.path.join(out, "summary.json"), "w"))
    time.sleep(0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main())
