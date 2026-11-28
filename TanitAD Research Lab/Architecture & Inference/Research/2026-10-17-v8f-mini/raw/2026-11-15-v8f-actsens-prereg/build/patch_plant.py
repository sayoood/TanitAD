import os
os.chdir("F:/TanitAD-scratch/actsens/box2_build/tests")
BS = chr(92)
p = "plant_trainer.py"
s = open(p, encoding="utf-8", newline="").read()
a = s.index("def good_selfcheck")
b = s.index("def main():")
new = '''def r35_terms(init):
    p = os.environ.get("ACTSENS_FAKE_R35_TERMS") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frozen", "r35_live_terms.json")
    return json.load(open(p, encoding="utf-8"))[f"s{init}"]["terms"]


def good_selfcheck(argv, mode="ok"):
    """the selfcheck.json a correct build writes for this argv; `mode` plants one defect (used by the dry run; the unit tests mutate the dict instead)"""
    F = parse(argv)
    one = lambda k, d=None: F[k][0] if F.get(k) else d                           # noqa: E731
    seed = int(one("seed", 0))
    init = int(one("init-ckpt").replace("BS", "/").split("M2b_s")[1][0])
    idm = [float(x) for x in F.get("wm-idm-gamma", [])]
    divrs = one("wm-sampler", "uniform") == "divrs"
    steps = int(one("steps", 0)) or 19164
    sha16 = hashlib.sha256(" ".join(argv).encode()).hexdigest()[:16]
    ish = json.load(open(os.environ["ACTSENS_FAKE_INIT_SHA_FILE"]))[str(init)]
    n = 2011970
    zc = "D:BStanitadBSzcacheBSM2b_s%dBStrain" % init
    lv = {"kin_mode": "yaw", "hmax": 91, "wm": "b", "wm_idm_gamma": 100.0, "wm_ch1_scale": 0.4, "w": {"w_wm": 0.1},
          "params": {"adapter": 0, "cond": 0, "S": 0, "T": 0, "O": 0, "WM": n}, "n_params_trainable": n,
          "wm_strides": [3, 5],
          "wm_only": {"wm_only": True, "wm_k": 1, "n_wm": n, "n_trainable": n},
          "wm_ladder_readback": {"hmax": 91, "built": {"k": [10, 10, 6], "ctx_built": [3, 3, 3], "cum_stride": [1, 3, 15], "states": [91, 31, 7], "anchors": [len(range(0, 90, 5)), len(range(0, 30, 5)), len(range(0, 6, 1))],
                                                          "ctx_passed": [3, 3, 3], "w_in_loss": [1.0, 1.0, 1.0]},
                                 "windows_built": {"kind": "wmonly", "F": 91, "stride": 1, "pooled": True, "t_min": 90, "zcache": zc, "zcache_missing": 0, "n_windows": 40358}},
          "recover": {"recover_p": 0.0}, "dagger_p": 0.0, "ego_perturb": {"ego_perturb_p": 0.0}}
    terms = dict(r35_terms(init))
    if idm:                                                                       # the IDM arm: unweighted terms equal R35's (same init, same first batch), level/L1 = R35's + (G1 - 100) x idm/L1
        terms["wm/level/L1"] = terms["wm/level/L1"] + (idm[0] - 100.0) * terms["wm/idm/L1"]
    smp = {"kind": "uniform", "class": "ChunkedSampler", "n_windows": 40358}
    sc = {"declared_vs_built": "PASS", "argv_sha": sha16, "steps_total": steps, "S_steps_per_epoch_100pct": 4791,
          "info": {"train_sha256": "37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1", "excluded_sha256": "f6a1dee02af0b96f815416bfef9bc11a04554702525135aa75a740942d7750f9", "n_clips": 4512, "n_excluded": 60,
                   "batch": 64, "md5_overridden": False, "excluded_in_train": 0, "n_windows": 40358, "train_window_list_sha256": "1e4fc4c4401f89dcb94d7981bfb4907ac18f87e82b77ebd1d3783e080ebf6e2c", "wm_sampler": smp,
                   "wm_only": {"path": one("init-ckpt"), "sha256": ish, "applied": True, "n_mismatch_after_load": 0, "mode": "wm_only", "source_arm": "M2b", "source_seed": init, "wm_reinit_seed": seed}},
          "levers": lv,
          "live": {"pass": True, "missing_terms": [], "dead_wm_groups": [], "unexpected_dead": [], "terms": terms, "control_adapter_grad_sum": 0.0, "control_nonwm_grad_sum": 0.0,
                   "wm_groups_grad": {"L1": 0.2, "L2": 0.1, "L3": 0.1}},
          "planner_identity_probe": {"pass": True, "n_tensors": 40, "n_equal": 40}}
    if idm:
        lv["wm_idm_gamma_readback"] = {"declared": idm, "built": idm, "wm_aux_gamma": 100.0}
    if divrs:
        sc["info"]["wm_sampler"] = {"kind": "divrs", "n_windows": 40358}
        lv["wm_sampler"] = {"kind": "divrs", "yaw_edges": [0.02, 0.05, 0.10, 0.20], "acc_edges": [-1.0, -0.3, 0.3, 1.0], "cap_mult": 10.0, "n_cells": 25, "n_windows": 40358, "cell_counts": [40358] + [0] * 24}
        sc["wm_sampler_probe"] = {"pass": True}
    if mode == "bad_idm_built" and idm:
        lv["wm_idm_gamma_readback"]["built"] = [100.0, 100.0, 100.0]
    elif mode == "bad_sampler" and divrs:                                        # the flag is set, the built sampler is the uniform one
        del lv["wm_sampler"]
        del sc["wm_sampler_probe"]
        sc["info"]["wm_sampler"] = smp
    elif mode == "bad_steps":
        sc["steps_total"] = 12345
    elif mode == "bad_zcache":
        lv["wm_ladder_readback"]["windows_built"]["zcache"] = "D:BStanitadBSzcacheBSM2b_s%dBStrain" % (1 - init)
    return sc


def model_state(seed):
    rng = np.random.RandomState(seed)
    keys = ["read_s.w", "head_s.w", "adapter.0.w", "pos", "time", "e_ego.w", "read_t.w", "trunk_t.w", "head_o.w", "read_o.w", "wm.stack.pred.w", "wm.proj.w"]
    return {k: rng.randn(4).astype(np.float32) for k in keys}


'''.replace("BS", BS * 2)
s = s[:a] + new + s[b:]
i = s.index("    every = 50 if steps <= 200 else 1000")
j = s.index("    time.sleep(0.05)")
tail = '''    every = 50 if steps <= 200 else 1000
    gam = [100.0, 100.0, 100.0]
    if F.get("wm-idm-gamma") and tm != "gamma_wrong":
        gam = [float(x) for x in F["wm-idm-gamma"]]

    def row(s):
        r = {"step": s, "sec": startup + sps * s}
        for L in (1, 2, 3):
            pred, sig, roll, asg, idm = 0.2, 3.0, 0.25, 8.0 if L >= 2 else 0.0, 0.11 * L
            r.update({f"wm/pred/L{L}": pred, f"wm/sigreg/L{L}": sig, f"wm/rollout/L{L}": roll, f"wm/idm/L{L}": idm})
            if L >= 2:
                r[f"wm/act_sigreg/L{L}"] = asg
            r[f"wm/level/L{L}"] = pred + roll + 0.08 * sig + gam[L - 1] * idm + (0.005 * asg if L >= 2 else 0.0)
        return r
    with open(os.path.join(out, "train_log.jsonl"), "w") as f:
        for s in sorted(set(range(every, steps + 1, every)) | {steps}):
            f.write(json.dumps(row(s)) + "\\n")
    init = int(one("init-ckpt").replace("BS", "/").split("M2b_s")[1][0])
    st = model_state(init)
    for k in list(st):
        if k.startswith("wm."):
            st[k] = st[k] + 1.0
    if tm == "move_frozen":
        st["e_ego.w"] = st["e_ego.w"] + 1.0
    n_opt = 12
    opt = {"param_groups": [{"params": list(range(n_opt))}], "state": {i: {"exp_avg": 1} for i in range(n_opt - (1 if tm == "no_adam_state" else 0))}}
    pickle.dump({"model": st, "step": steps, "opt": opt}, open(os.path.join(out, "ckpt_last.pt"), "wb"))
    json.dump({"done": True, "steps": steps, "arm": one("arm"), "seed": int(one("seed", 0)), "argv_sha": sc["argv_sha"]}, open(os.path.join(out, "summary.json"), "w"))
'''.replace("BS", BS * 2)
s = s[:i] + tail + s[j:]
s = s.replace('"--wm-speed" def bad_wm_speed def wm_speed_readback wm_speed=a.wm_speed with_v"""', '"--wm-idm-gamma" "--wm-sampler" def bad_wm_idm_gamma def bad_wm_sampler lever:wm_idm_gamma data:wm_sampler def make_sampler def wm_sampler_probe wm_gamma_levels class DiversitySampler def window_act_means def check_cell_shares class V8FMiniWMWindows"""')
open(p, "w", encoding="utf-8", newline="").write(s)

p = "dry_helpers.py"
s = open(p, encoding="utf-8", newline="").read()
old = 'CK.MODEL_REL: "# def built_speed  def ego_stack_cls  reflect_integrate  anchored_speeds\\n", CK.DATA_REL: "# with_v\\n", CK.WINDOWS_REL: "# self.ego\\n",'
assert old in s
s = s.replace(old, 'CK.MODEL_REL: "# wm_gamma_levels\\n", CK.DATA_REL: "# class DiversitySampler  def window_act_means  def check_cell_shares\\n", CK.WINDOWS_REL: "# class V8FMiniWMWindows\\n",')
open(p, "w", encoding="utf-8", newline="").write(s)
print("patched")
