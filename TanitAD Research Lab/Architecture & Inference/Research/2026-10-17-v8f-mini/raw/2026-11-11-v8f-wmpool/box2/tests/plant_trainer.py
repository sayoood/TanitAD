"""Dry-run STAND-IN for stack/scripts/v8f_mini_train.py (dev box only; shipped through a throw-away git repo by the dry run). It writes what the trainer at abc01481 is documented to write
(selfcheck.json with the keys wmpool_check.gate_problems reads, train_log.jsonl with cumulative `sec`, summary.json, a pickled ckpt_last.pt) from its OWN reading of the argv, steered by the file in PLANT_CONTROL (json).
Like the real trainer it refuses (rc 2, ZZDVB-REFUSEDZZ) a --wm-zcache without CELL_MANIFEST.json, and a manifest whose adapter sha256 is not the init checkpoint's seed's (the stand-in's adapter sha is sha256 of "adapter-<seed>").
Needle strings the ship step requires, present here so the stand-in passes it: "--wm-pool" def bad_wm_pool def wm_pool_readback wm_pool=a.wm_pool pool_cache_problems"""
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

BS = chr(92)


def parse(argv):
    F, cur = {}, None
    for t in argv:
        if t.startswith("--"):
            cur = t[2:]
            F[cur] = []
        elif cur:
            F[cur].append(t)
    return F


def adapter_sha(seed):
    return hashlib.sha256(("adapter-%d" % seed).encode()).hexdigest()


def seed_of_init(path):
    return int(path.replace(BS, "/").split("M2b_s")[1][0])


def good_selfcheck(argv, mode="ok"):
    """the selfcheck.json a correct build writes for this argv; `mode` plants one defect (used by the dry run; the unit tests mutate the dict instead)"""
    F = parse(argv)
    one = lambda k, d=None: F[k][0] if F.get(k) else d                           # noqa: E731
    seed = int(one("seed", 0))
    init = seed_of_init(one("init-ckpt"))
    steps = int(one("steps", 0)) or 19164
    sha16 = hashlib.sha256(" ".join(argv).encode()).hexdigest()[:16]
    ish = json.load(open(os.environ["WMPOOL_FAKE_INIT_SHA_FILE"]))[str(init)]
    n = 2143298 + 256
    zc = "D:" + BS + "tanitad" + BS + "zcells" + BS + "M2b_s%d" % init + BS + "train"
    lv = {"kin_mode": "yaw", "hmax": 91, "wm": "b", "wm_idm_gamma": 100.0, "wm_ch1_scale": 0.4, "w": {"w_wm": 0.1},
          "params": {"adapter": 0, "cond": 0, "S": 0, "T": 0, "O": 0, "WM": n}, "n_params_trainable": n,
          "wm_k": 3, "wm_k_levels": [3, 3, 3], "wm_strides": [3, 5],
          "wm_only": {"wm_only": True, "wm_k": 3, "n_wm": n, "n_trainable": n},
          "wm_pool": "att25",
          "wm_pool_readback": {"declared": "att25", "built": {"mode": "att25", "n_params": 256, "q_shape": [256], "q_zero_at_build": True, "n_cells": 10, "cell_map_ok": True, "uniform_at_build": True, "attends_when_set": True},
                               "dataset_pool": "att25", "dataset_zcell_shape": [91, 10, 256], "zcache": zc, "cache_problems": []},
          "wm_ladder_readback": {"hmax": 91, "built": {"k": [10, 10, 6], "ctx_built": [3, 3, 3], "cum_stride": [1, 3, 15], "states": [91, 31, 7], "anchors": [len(range(0, 90, 5)), len(range(0, 30, 5)), len(range(0, 6, 1))],
                                                          "ctx_passed": [3, 3, 3], "w_in_loss": [1.0, 1.0, 1.0]},
                                 "windows_built": {"kind": "wmonly", "F": 91, "stride": 1, "pooled": True, "t_min": 90, "zcache": zc, "zcache_missing": 0, "n_windows": 1000}},
          "recover": {"recover_p": 0.0}, "dagger_p": 0.0, "ego_perturb": {"ego_perturb_p": 0.0}}
    terms = {"wm": 1.0, "wm/rollout/L1": 0.3, "total": 1.0}
    sc = {"declared_vs_built": "PASS", "argv_sha": sha16, "steps_total": steps, "S_steps_per_epoch_100pct": 4791,
          "info": {"train_sha256": "37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1", "excluded_sha256": "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9", "n_clips": 4512, "n_excluded": 60,
                   "batch": 64, "md5_overridden": False, "excluded_in_train": 0,
                   "wm_only": {"path": one("init-ckpt"), "sha256": ish, "applied": True, "n_mismatch_after_load": 0, "mode": "wm_only", "source_arm": "M2b", "source_seed": init, "wm_reinit_seed": seed}},
          "levers": lv,
          "live": {"pass": True, "missing_terms": [], "dead_wm_groups": [], "unexpected_dead": [], "terms": terms, "control_adapter_grad_sum": 0.0, "control_nonwm_grad_sum": 0.0,
                   "wm_groups_grad": {"L1": 0.2, "L2": 0.1, "L3": 0.1}},
          "planner_identity_probe": {"pass": True, "n_tensors": 40, "n_equal": 40}}
    pb = lv["wm_pool_readback"]["built"]
    if mode == "bad_pool_mode":
        pb["mode"] = "mean"
    elif mode == "query_not_zero":
        pb["q_zero_at_build"] = False
    elif mode == "not_uniform":
        pb["uniform_at_build"] = False
    elif mode == "no_attention":
        pb["attends_when_set"] = False
    elif mode == "bad_cell_map":
        pb["cell_map_ok"] = False
    elif mode == "no_pool_readback":
        lv.pop("wm_pool_readback")
    elif mode == "no_pool_lever":
        lv.pop("wm_pool")
    elif mode == "bad_cell_shape":
        lv["wm_pool_readback"]["dataset_zcell_shape"] = [91, 40, 256]
    elif mode == "cache_problem":
        lv["wm_pool_readback"]["cache_problems"] = ["--wm-pool att25: the cell cache was built from adapter 123456789abc, the built model's adapter is fedcba987654"]
    elif mode == "bad_steps":
        sc["steps_total"] = 12345
    elif mode == "bad_zcache":
        lv["wm_ladder_readback"]["windows_built"]["zcache"] = "D:" + BS + "tanitad" + BS + "zcells" + BS + "M2b_s%d" % (1 - init) + BS + "train"
    elif mode == "mean_cache":
        lv["wm_ladder_readback"]["windows_built"]["zcache"] = "D:" + BS + "tanitad" + BS + "zcache" + BS + "M2b_s%d" % init + BS + "train"
    return sc


def model_state(seed):
    rng = np.random.RandomState(seed)
    keys = ["read_s.w", "head_s.w", "adapter.0.w", "pos", "time", "e_ego.w", "read_t.w", "trunk_t.w", "head_o.w", "read_o.w", "wm.stack.pred.w", "wm.proj.w"]
    return {k: rng.randn(4).astype(np.float32) for k in keys}


def main():
    argv = sys.argv[1:]
    F = parse(argv)
    one = lambda k, d=None: F[k][0] if F.get(k) else d                           # noqa: E731
    ctl = json.load(open(os.environ["PLANT_CONTROL"])) if os.environ.get("PLANT_CONTROL") and os.path.exists(os.environ["PLANT_CONTROL"]) else {}
    root = os.environ["WMPOOL_ROOT"].replace(BS, "/")

    def mapped(p):
        p = p.replace(BS, "/")
        return root + p[len("d:/tanitad"):] if p.lower().startswith("d:/tanitad") else p
    out = mapped(one("out"))
    init = seed_of_init(one("init-ckpt"))
    man = os.path.join(mapped(one("wm-zcache", "")), "CELL_MANIFEST.json")           # the trainer's own early refusal (pool_cache_problems)
    if not os.path.isfile(man):
        print("ZZDVB-REFUSEDZZ --wm-pool att25: no CELL_MANIFEST.json in " + os.path.dirname(man))
        return 2
    if json.load(open(man)).get("adapter_state_sha256") != adapter_sha(init):
        print("ZZDVB-REFUSEDZZ --wm-pool att25: the cell cache was built from another adapter than the built model's")
        return 2
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
    st = model_state(init)
    for k in list(st):
        if k.startswith("wm."):
            st[k] = st[k] + 1.0
    if tm != "no_pool_q":
        st["wm.pool_q"] = np.zeros(4, dtype=np.float32) if tm == "no_pool_move" else np.full(4, 0.01, dtype=np.float32)
    if tm == "move_frozen":
        st["e_ego.w"] = st["e_ego.w"] + 1.0
    pickle.dump({"model": st, "step": steps}, open(os.path.join(out, "ckpt_last.pt"), "wb"))
    json.dump({"done": True, "steps": steps, "arm": one("arm"), "seed": int(one("seed", 0)), "argv_sha": sc["argv_sha"]}, open(os.path.join(out, "summary.json"), "w"))
    time.sleep(0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main())
