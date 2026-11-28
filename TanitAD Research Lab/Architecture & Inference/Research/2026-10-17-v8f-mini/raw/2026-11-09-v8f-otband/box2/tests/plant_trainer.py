"""Dry-run STAND-IN for stack/scripts/v8f_mini_train.py (dev box only; shipped through a throw-away git repo by the dry run). It writes what the trainer at 8e9499e0 is documented to write
(selfcheck.json with the keys otband_check.gate_problems reads, train_log.jsonl with cumulative `sec`, summary.json, a pickled ckpt_last.pt) from its OWN reading of the argv, steered by the file in PLANT_CONTROL (json).
Needle strings the ship step requires, present here so the stand-in passes it: "--w-op" "--w-tac-traj" "--t-stok-dir" "--freeze" def bad_oband def oband_probe def frozen_groups FREEZE_GROUPS"""
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np


def parse(argv):
    F, cur = {}, None
    for t in argv:
        if t.startswith("--"):
            cur = t[2:]
            F[cur] = []
        elif cur:
            F[cur].append(t)
    return F


def good_selfcheck(argv, mode="ok"):
    """the selfcheck.json a correct build writes for this argv; `mode` plants one defect (used by the dry run; the unit tests mutate the dict instead)"""
    F = parse(argv)
    one = lambda k, d=None: F[k][0] if F.get(k) else d                           # noqa: E731
    seed = int(one("seed", 0))
    init = int(one("init-ckpt").replace("\\", "/").split("M2b_s")[1][0])
    wop, wtac = float(one("w-op", 0.0)), float(one("w-tac-traj", 0.0))
    tok = bool(F.get("t-stok-dir"))
    band = wop > 0 or wtac > 0
    steps = int(one("steps", 0)) or 4791 * int(one("steps-mult", 1))
    sha16 = hashlib.sha256(" ".join(argv).encode()).hexdigest()[:16]
    ish = json.load(open(os.environ["OTBAND_FAKE_INIT_SHA_FILE"]))[str(init)]
    lv = {"kin_mode": "yaw", "params": {"adapter": 0, "cond": 0, "S": 0, "T": 1000, "O": 2000, "WM": 3000}, "n_params_trainable": 6000,
          "recover": {"recover_p": 0.0}, "dagger_p": 0.0, "ego_perturb": {"ego_perturb_p": 0.0}}
    terms = {"ckpt": 1.0, "goal": 1.0, "lat": 1.0, "lon": 1.0, "cons": 1.0, "ade": 2.0, "fde": 3.0, "acc": 0.1, "total": 5.0, "wm": 1.0}
    gg = {"adapter": 0.0, "S": 0.0, "T": 1.5, "O": 2.0, "head_s": 0.0, "head_o": 2.0, "wm": 1.0}
    sc = {"declared_vs_built": "PASS", "argv_sha": sha16, "steps_total": steps, "S_steps_per_epoch_100pct": 4791,
          "info": {"train_sha256": "37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1", "excluded_sha256": "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9", "n_clips": 4512, "n_excluded": 60,
                   "batch": 64, "md5_overridden": False, "excluded_in_train": 0,
                   "init_ckpt": {"path": one("init-ckpt"), "sha256": ish, "applied": True, "n_mismatch_after_load": 0, "n_tensors": 40, "source_arm": "M2b", "source_seed": init}},
          "levers": lv, "live": {"pass": True, "missing_terms": [], "unexpected_dead": [], "terms": terms, "control_prior_grad_sum": 0.0, "head_o_grad_step0": 0.7, "groups_grad": gg}}
    if band:
        lv["w_op"], lv["w_tac_traj"] = wop, wtac
        terms.update({"ade_op": 1.0, "fde_op": 1.2, "ade_tac": 3.0, "fde_tac": 4.0, "op": 0.5, "tac": 1.5})
        sc["otband_probe"] = {"otband": "on", "stok_dim": 8 if tok else 0, "detach_upper": True,
                              "contribution": {"w_op": {"weight": wop, "term": 0.5, "moved_total_by": wop * 0.5}, "w_tac_traj": {"weight": wtac, "term": 1.5, "moved_total_by": wtac * 1.5}},
                              "purity": {"upper_grad_abs_sum": 0.0, "head_o": 0.3, "by_term": {"op": {"upper_grad_abs_sum": 0.0, "head_o": 0.3}, "tac": {"upper_grad_abs_sum": 0.0, "head_o": 0.4}}}, "code_edge_max_dknot": 0.2, "pass": True}
    else:
        sc["otband_probe"] = {"otband": "detach-only", "stok_dim": 0, "detach_upper": True,
                              "purity": {"upper_grad_abs_sum": 0.0, "head_o": 0.3, "by_term": {"ade": {"upper_grad_abs_sum": 0.0, "head_o": 0.3}, "fde": {"upper_grad_abs_sum": 0.0, "head_o": 0.4}}}, "code_edge_max_dknot": 0.2, "pass": True}
    if tok:
        lv["stok_dim"] = 8
        lv["params"]["STOK"] = 9
        sc["info"]["init_ckpt"]["declared_new_keys"] = ["e_stok.weight", "e_stok.bias"]
        sc["otband_probe"]["token_edge"] = {"max_dlogit": 0.1, "max_dtraj30": 0.1}
        sc["stok"] = {"stok_dim": 8, "stok_shuffle": -1}
    if mode == "bad_freeze":
        lv["params"]["S"] = 77
    elif mode == "bad_detach" and band:
        sc["otband_probe"]["purity"]["upper_grad_abs_sum"] = 1e-9
    elif mode == "bad_steps":
        sc["steps_total"] = 12345
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
    root = os.environ["OTBAND_ROOT"].replace("\\", "/")
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
    every = 50 if steps <= 200 else 1000
    with open(os.path.join(out, "train_log.jsonl"), "w") as f:
        for s in sorted(set(range(every, steps + 1, every)) | {steps}):
            f.write(json.dumps({"step": s, "sec": startup + sps * s}) + "\n")
    init = int(one("init-ckpt").replace("\\", "/").split("M2b_s")[1][0])
    st = model_state(init)
    if tm != "no_move":
        for k in st:
            if k.startswith(("head_o", "read_o", "trunk_t", "h_goal")):
                st[k] = st[k] + 1.0
    if tm == "move_frozen":
        st["e_ego.w"] = st["e_ego.w"] + 1.0
    wop, wtac = float(one("w-op", 0.0)), float(one("w-tac-traj", 0.0))
    cfg = {"detach_upper": True, "w_traj": 0.0 if (wop or wtac) else 1.0, "w_op": wop, "w_tac_traj": wtac, "stok_dim": 8 if F.get("t-stok-dir") else 0}
    pickle.dump({"model": st, "cfg": cfg, "step": steps}, open(os.path.join(out, "ckpt_last.pt"), "wb"))
    json.dump({"done": True, "steps": steps, "arm": one("arm"), "seed": int(one("seed", 0)), "argv_sha": sc["argv_sha"]}, open(os.path.join(out, "summary.json"), "w"))
    time.sleep(0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main())
