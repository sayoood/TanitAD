"""Dry-run STAND-IN for stack/scripts/v8f_mini_train.py (dev box only; shipped through a throw-away git repo by the dry run). It writes what the trainer at the code of record (2bca1c52) is documented to write
(selfcheck.json with the keys stopbal_check.gate_problems reads, train_log.jsonl with cumulative `sec`, summary.json, a pickled ckpt_last.pt) from its OWN reading of the argv, steered by the file in PLANT_CONTROL (json).
Needle strings the ship step requires, present here so the stand-in passes it: "--goal-pos-weight-stop" "--lon-class-weight-stop" goal_pos_weight_stop lon_class_weight_stop "--w-acc" "--dagger-p" def acc_probe def dagger_probe"""
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

PARAMS = {"adapter": 275712, "cond": 25088, "S": 1321737, "T": 1463868, "O": 2120216, "WM": 2007874}


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


def good_selfcheck(argv, mode="ok"):
    """the selfcheck.json a correct build writes for this argv; `mode` plants one defect (used by the dry run; the unit tests mutate the dict instead)"""
    F = parse(argv)
    one = lambda k, d=None: F[k][0] if F.get(k) else d                           # noqa: E731
    init = int(one("init-ckpt").replace("\\", "/").split("M2b_s")[1][0])
    wg, wl = float(one("goal-pos-weight-stop", 1.0)), float(one("lon-class-weight-stop", 1.0))
    wacc = float(one("w-acc", 0.0))
    steps = int(one("steps", 0)) or 4791 * int(one("steps-mult", 1))
    sha16 = hashlib.sha256(" ".join(argv).encode()).hexdigest()[:16]
    ish = json.load(open(os.environ["STOPBAL_FAKE_INIT_SHA_FILE"]))[str(init)]
    bsha = bank_sha_of(init)
    lv = {"kin_mode": "yaw", "params": dict(PARAMS), "n_params_trainable": 7214495, "w_acc": wacc, "goal_pos_weight_stop": wg, "lon_class_weight_stop": wl, "p_gt_start": 0.2, "p_gt_end": 0.2,
          "recover": {"recover_p": 0.0}, "dagger_p": float(one("dagger-p", 0.0)), "dagger_bank_sha256": bsha, "dagger_bank_sha256_declared": one("dagger-bank-sha256"),
          "dagger": {"dagger_p": float(one("dagger-p", 0.0)), "dagger_seed": int(one("dagger-seed", 0)), "dagger_bank": one("dagger-bank") + "\\bank.json", "dagger_bank_sha256": bsha},
          "ego_perturb": {"ego_perturb_p": 0.0}, "ego_perturb_with_dagger": False, "launch_intent": {"launch_intent_w": 0.0}}
    terms = {"ckpt": 1.0, "goal": 1.0, "lat": 1.0, "lon": 1.0, "cons": 1.0, "ade": 2.0, "fde": 3.0, "acc": 0.1, "total": 5.0, "wm": 1.0}
    gg = {"adapter": 9.0, "S": 3.0, "T": 1.5, "O": 2.0, "head_s": 1.0, "head_o": 2.0, "wm": 1.0}
    sc = {"declared_vs_built": "PASS", "argv_sha": sha16, "steps_total": steps, "S_steps_per_epoch_100pct": 4791,
          "info": {"train_sha256": "37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1", "excluded_sha256": "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9", "n_clips": 4512, "n_excluded": 60,
                   "batch": 64, "md5_overridden": False, "excluded_in_train": 0,
                   "init_ckpt": {"path": one("init-ckpt"), "sha256": ish, "applied": True, "n_mismatch_after_load": 0, "n_tensors": 40, "source_arm": "M2b", "source_seed": init},
                   "dagger": {"bank_sha256": bsha, "coverage": {"blocked": False, "fraction": 0.75}}},
          "levers": lv, "live": {"pass": True, "missing_terms": [], "dead_params": [], "unexpected_dead": [], "terms": terms, "control_prior_grad_sum": 0.0, "head_o_grad_step0": 0.7, "groups_grad": gg},
          "acc_probe": {"w_acc_declared": wacc, "w_probe": wacc, "acc": 0.01, "contribution": wacc * 0.01, "w_x_acc": wacc * 0.01, "head_o_grad_diff": 4.0, "read_o_grad_diff": 8.0,
                        "control_repeat_total_diff": 0.0, "control_repeat_grad_diff": 0.0, "control_prior_grad_diff": 0.0, "pass": True},
          "dagger_probe": {"p_declared": float(one("dagger-p", 0.0)), "n_rows": 64, "n_eligible": 49, "n_selected_at_p1": 49, "control_repeat_loss_diff": 0.0, "control_repeat_grad_diff": 0.0,
                           "control_p0_loss_diff": 0.0, "control_p0_grad_diff": 0.0, "control_prior_grad_diff": 0.0, "pass": True}}
    if mode == "bad_params":
        lv["params"]["S"] = 0
    elif mode == "lever_not_built":                                              # the arm's own lever reads back 1.0: the weight never reached the built loss
        lv["goal_pos_weight_stop"], lv["lon_class_weight_stop"] = 1.0, 1.0
    elif mode == "dead_goal_head":
        sc["live"]["dead_params"] = ["h_goal.0.weight"]
        sc["live"]["unexpected_dead"] = ["h_goal.0.weight"]
    elif mode == "bad_steps":
        sc["steps_total"] = 12345
    elif mode == "bad_bank":
        lv["dagger_bank_sha256"] = "0" * 64
    return sc


def model_state(seed):
    rng = np.random.RandomState(seed)
    keys = ["read_s.w", "head_s.w", "adapter.0.w", "pos", "time", "e_ego.w", "e_nav.w", "e_lim.w", "e_age.w", "e_ck.w", "e_code.w", "read_t.w", "trunk_t.w", "h_goal.w", "head_o.w", "read_o.w"]
    return {k: rng.randn(4).astype(np.float32) for k in keys}


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
    if "selfcheck-only" in F:
        mode = (ctl.get("modes") or {}).get(os.path.basename(out), ctl.get("mode", "ok"))        # per-tag defects: the control file maps a gate out dir name to a mode
        json.dump(good_selfcheck(argv, mode), open(os.path.join(out, "selfcheck.json"), "w"), indent=1)
        return 0
    tm = (ctl.get("train_modes") or {}).get(os.path.basename(out), ctl.get("train_mode"))
    if tm == "crash":
        return 1
    sc = good_selfcheck(argv)
    steps = sc["steps_total"]
    sps, startup = float(ctl.get("sps", 1.0)), float(ctl.get("startup_s", 30.0))
    every = 50 if steps <= 200 else 500
    with open(os.path.join(out, "train_log.jsonl"), "w") as f:
        for s in sorted(set(range(every, steps + 1, every)) | {steps}):
            f.write(json.dumps({"step": s, "sec": startup + sps * s}) + "\n")
    init = int(one("init-ckpt").replace("\\", "/").split("M2b_s")[1][0])
    seed = int(one("seed", 0))
    st = model_state(init)
    moves = {"S": ("read_s", "head_s"), "T": ("read_t", "trunk_t", "h_goal"), "O": ("read_o", "head_o")}
    for g, pre in moves.items():
        if tm == "no_move" or (tm == "S_frozen" and g == "S"):
            continue
        for k in st:
            if k.startswith(pre):
                st[k] = st[k] + 1.0 + 0.01 * seed
    cfg = {"detach_upper": False, "w_traj": 1.0, "w_acc": float(one("w-acc", 0.0)), "goal_pos_weight_stop": float(one("goal-pos-weight-stop", 1.0)), "lon_class_weight_stop": float(one("lon-class-weight-stop", 1.0))}
    if tm == "cfg_weight_lost":                                                  # the checkpoint's config does not carry the declared weights
        cfg["goal_pos_weight_stop"] = cfg["lon_class_weight_stop"] = 1.0
    dg = {"dagger_p": float(one("dagger-p", 0.0)), "dagger_seed": int(one("dagger-seed", 0)), "dagger_bank_sha256": bank_sha_of(init)}
    ck_seed = seed + 1 if tm == "wrong_seed" else seed
    # the checkpoint embeds argv_sha (which holds --out): two arms with different --out can never share a file sha256, as in the real trainer (v8f_mini_train.py save())
    pickle.dump({"model": st, "cfg": cfg, "step": steps, "seed": ck_seed, "arm": one("arm"), "argv_sha": sc["argv_sha"], "dagger": dg,
                 "init_ckpt": {"sha256": sc["info"]["init_ckpt"]["sha256"]}}, open(os.path.join(out, "ckpt_last.pt"), "wb"))
    json.dump({"done": True, "steps": steps, "arm": one("arm"), "seed": seed, "argv_sha": sc["argv_sha"]}, open(os.path.join(out, "summary.json"), "w"))
    time.sleep(0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main())
