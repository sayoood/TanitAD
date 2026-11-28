"""Unit tests of optb_check.py (ACTRSP Option B) (pure Python, no torch, no GPU, no network). Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I tests/test_actrsp_check.py  -> prints  UNIT ok=<n> bad=<m>  (exit 1 on any bad)
Every gate assertion on the response lever has a planted-bad selfcheck that must be refused; every clean selfcheck must pass; an argv whose --wm-rsp-w is not named must be refused."""
import copy
import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
import optb_check as CK  # noqa: E402

N_OK, N_BAD = 0, 0


def check(name, cond):
    global N_OK, N_BAD
    if cond:
        N_OK += 1
    else:
        N_BAD += 1
        print(f"UNIT-FAIL {name}")


def good_selfcheck(tag):
    """what a correct build writes for this tag's gate argv (the keys gate_problems reads)"""
    toks = CK.tokens(tag)
    gt = CK.gate_argv(tag, toks)
    spec = CK.TABLE[CK.base_tag(tag)]
    init, seed, rsp = spec["init"], spec["seed"], spec["lever"] == "rsp"
    n = CK.WM_PARAMS_BASE
    nt = n - 4000      # trainable = WM minus wm.proj (exact count UNVERIFIED until the pilot selfcheck)
    zc = "D:\\tanitad\\zcache\\M2b_s%d\\train" % init
    ic = "D:\\tanitad\\optb\\inputs\\ladder_R35_s%d.pt" % init
    terms = {"wm": 1.0, "total": 0.1}
    for L in (1, 2, 3):
        for t in ("pred", "sigreg", "idm", "level", "rollout", "rollout_valid", "rollout_masked"):
            terms[f"wm/{t}/L{L}"] = 0.5 + 0.01 * L
        if L >= 2:
            terms[f"wm/act_sigreg/L{L}"] = 0.3
    lv = {"kin_mode": "yaw", "hmax": 91, "wm": "b", "wm_idm_gamma": 100.0, "wm_ch1_scale": 0.4, "w": {"w_wm": 0.1}, "params": {"adapter": 0, "cond": 0, "S": 0, "T": 0, "O": 0, "WM": nt}, "n_params_trainable": nt,
          "wm_strides": [3, 5], "wm_only": {"wm_only": True, "wm_k": 1, "n_wm": n, "n_trainable": nt},
          "wm_ladder_readback": {"hmax": 91, "built": {"k": [10, 10, 6], "ctx_built": [3, 3, 3], "cum_stride": [1, 3, 15], "states": [91, 31, 7], "anchors": [len(range(0, 90, 5)), len(range(0, 30, 5)), len(range(0, 6, 1))],
                                                          "ctx_passed": [3, 3, 3], "w_in_loss": [1.0, 1.0, 1.0]},
                                 "windows_built": {"kind": "wmonly", "F": 91, "stride": 1, "pooled": True, "t_min": 90, "zcache": zc, "zcache_missing": 0, "n_windows": CK.N_WINDOWS}},
          "recover": {"recover_p": 0.0}, "dagger_p": 0.0, "ego_perturb": {"ego_perturb_p": 0.0}}
    live = {"pass": True, "missing_terms": [], "dead_wm_groups": [], "unexpected_dead": [], "terms": terms, "control_adapter_grad_sum": 0.0, "control_nonwm_grad_sum": 0.0, "wm_groups_grad": {"L1": 0.2, "L2": 0.1, "L3": 0.1}}
    sc = {"declared_vs_built": "PASS", "argv_sha": CK.argv_sha16(gt), "steps_total": CK.RUN_STEPS, "S_steps_per_epoch_100pct": 4791,
          "info": {"train_sha256": CK.DATA["train_sha256"], "excluded_sha256": CK.DATA["excluded_sha256"], "n_clips": 4512, "n_excluded": 60, "batch": 64, "md5_overridden": False, "excluded_in_train": 0, "n_windows": CK.N_WINDOWS,
                   "train_window_list_sha256": CK.WINDOW_LIST_SHA, "wm_sampler": {"kind": "uniform", "class": "ChunkedSampler", "n_windows": CK.N_WINDOWS},
                   "wm_only": {"path": ic, "sha256": CK.INIT_SHA[init], "applied": True, "n_mismatch_after_load": 0, "mode": "wm_only_init_wm", "init_wm": True}},
          "levers": lv, "live": live, "planner_identity_probe": {"pass": True, "n_tensors": 40, "n_equal": 40}}
    lv["wm_rsp_readback"] = {"declared": 0.0, "built_config": 0.0, "hmax": 91, "declared_readout": "idm", "built_readout": "idm", "built": {"w_in_loss_rsp": 0.0},
                             "freeze_proj_declared": "on", "wm_proj_requires_grad": [False], "init_from_ckpt_wm_declared": "on"}
    if rsp:
        terms["wm/rsp/L1"] = 0.4
        first = dict(terms)
        want = 1.0 * terms["wm/rsp/L1"]
        live["c3"] = {"n_unweighted": 20, "unweighted_equal": True, "level_L1_diff": want, "rsp_w_x_rsp": want, "levels_23_equal": True, "ref_has_no_rsp_keys": True, "rel_err": 0.0, "tol": 1e-4, "pass": True}
        sc["live_first"] = first
        lv["wm_rsp_readback"].update({"declared": 1.0, "built_config": 1.0, "declared_readout": "fixed_k", "built_readout": "fixed_k", "built": {"calls": 1, "contrasts": 2, "anchors": 9, "k": 10, "ctx_passed": 1, "w_in_loss_rsp": 1.0, "rsp_term": 0.4, "readout": "fixed_k", "s_cal": CK.S_CAL[init]}})
    return toks, sc


def planted(tag, fn):
    toks, sc = good_selfcheck(tag)
    sc2 = copy.deepcopy(sc)
    fn(sc2)
    return CK.gate_problems(tag, toks, sc2)


def main():
    check("run order: chassis 0 (FTRSP_s0, FTNULL_s0, FTRSP_s2, FTNULL_s2, FTNULL_s4) then chassis 1", CK.RUN_ORDER == ["FTRSP_s0", "FTNULL_s0", "FTRSP_s2", "FTNULL_s2", "FTNULL_s4", "FTRSP_s1", "FTNULL_s1", "FTRSP_s3", "FTNULL_s3", "FTNULL_s5"])
    check("ten arms", len(CK.TABLE) == 10)
    for t in CK.TABLE:
        toks = CK.tokens(t)
        check(f"frozen argv of {t} is accepted", CK.audit_line(t, " ".join(toks)) == [])
        check(f"{t}: argv_rsp_problems clean", CK.argv_rsp_problems(t, toks) == [])
        lev = CK.TABLE[t]["lever"]
        check(f"{t}: --wm-rsp-w named exactly {'once as 1.0' if lev == 'rsp' else 'never'}", toks.count("--wm-rsp-w") == (1 if lev == "rsp" else 0))
        check(f"{t}: no ACTSENS lever flag", "--wm-idm-gamma" not in toks and "--wm-sampler" not in toks)
        check(f"{t}: freeze + init named on", toks[toks.index("--wm-freeze-proj") + 1] == "on" and toks[toks.index("--wm-init-from-ckpt-wm") + 1] == "on")
        check(f"{t}: RSP readout flags named only on RSP arms", all((f in toks) == (lev == "rsp") for f in ("--wm-rsp-readout", "--wm-rsp-probe", "--wm-rsp-probe-sha256")))
        toks_g, sc = good_selfcheck(t)
        check(f"{t}: the clean selfcheck passes the gate", CK.gate_problems(t, toks_g, sc) == [])
    # the argv rule
    t = CK.tokens("FTRSP_s0")
    unnamed = [x for i, x in enumerate(t) if not (x == "--wm-rsp-w" or (i > 0 and t[i - 1] == "--wm-rsp-w"))]
    check("an RSP argv without --wm-rsp-w is refused", CK.argv_rsp_problems("FTRSP_s0", unnamed) != [])
    check("an RSP argv with --wm-rsp-w 0.3 is refused", CK.argv_rsp_problems("FTRSP_s0", [("0.3" if (i > 0 and t[i - 1] == "--wm-rsp-w") else x) for i, x in enumerate(t)]) != [])
    check("an RSP argv naming the flag twice is refused", CK.argv_rsp_problems("FTRSP_s0", t + ["--wm-rsp-w", "1.0"]) != [])
    check("a NULL argv that names the flag is refused", CK.argv_rsp_problems("FTNULL_s4", CK.tokens("FTNULL_s4") + ["--wm-rsp-w", "1.0"]) != [])
    check("the gate refuses an unnamed RSP argv with the problem named", any("--wm-rsp-w" in x for x in CK.gate_problems("FTRSP_s0", unnamed, good_selfcheck("FTRSP_s0")[1])))
    check("audit_line refuses a shipped RSP line without the flag", CK.audit_line("FTRSP_s0", " ".join(unnamed)) != [])
    # planted defects on the response lever
    def rb(sc, **kw):
        sc["levers"]["wm_rsp_readback"].update(kw)
    check("missing wm_rsp_readback is refused", planted("FTRSP_s0", lambda sc: sc["levers"].pop("wm_rsp_readback")) != [])
    check("built_config 0.0 (a builder that ignores the weight) is refused", planted("FTRSP_s0", lambda sc: rb(sc, built_config=0.0)) != [])
    check("built loss weight 0.0 (computed, not added) is refused", planted("FTRSP_s0", lambda sc: sc["levers"]["wm_rsp_readback"]["built"].update(w_in_loss_rsp=0.0)) != [])
    check("built loss weight 1.0000005 is accepted, 1.00001 refused", planted("FTRSP_s0", lambda sc: sc["levers"]["wm_rsp_readback"]["built"].update(w_in_loss_rsp=1.0000005)) == [] and planted("FTRSP_s0", lambda sc: sc["levers"]["wm_rsp_readback"]["built"].update(w_in_loss_rsp=1.00001)) != [])
    for k, v in (("anchors", 8), ("k", 5), ("contrasts", 4), ("ctx_passed", 3), ("calls", 2), ("readout", "idm")):
        check(f"geometry {k} = {v} is refused", planted("FTRSP_s0", lambda sc, k=k, v=v: sc["levers"]["wm_rsp_readback"]["built"].update({k: v})) != [])
    check("rsp term missing from the live terms is refused", planted("FTRSP_s0", lambda sc: sc["live"]["terms"].pop("wm/rsp/L1")) != [])
    check("live_first missing is refused", planted("FTRSP_s0", lambda sc: sc.pop("live_first")) != [])
    check("live.c3 missing is refused", planted("FTRSP_s0", lambda sc: sc["live"].pop("c3")) != [])
    check("C3 not passed is refused", planted("FTRSP_s0", lambda sc: sc["live"]["c3"].update(pass_=False, **{"pass": False})) != [])
    check("C3 level difference zero (term not added) is refused", planted("FTRSP_s0", lambda sc: sc["live"]["c3"].update(level_L1_diff=0.0)) != [])
    check("C3 unweighted terms differ is refused", planted("FTRSP_s0", lambda sc: sc["live"]["c3"].update(unweighted_equal=False)) != [])
    check("a NULL arm WITHOUT wm_rsp_readback is refused (it names freeze + init)", planted("FTNULL_s4", lambda sc: sc["levers"].pop("wm_rsp_readback")) != [])
    check("a NULL arm whose readback shows the fixed_k readout is refused", planted("FTNULL_s4", lambda sc: sc["levers"]["wm_rsp_readback"].update(built_readout="fixed_k")) != [])
    check("a NULL arm carrying a response term is refused", planted("FTNULL_s4", lambda sc: sc["live"]["terms"].update({"wm/rsp/L1": 0.4})) != [])
    check("an ACTSENS lever readback on an RSP arm is refused", planted("FTRSP_s0", lambda sc: sc["levers"].update(wm_idm_gamma_readback={"declared": [1000.0, 100.0, 100.0], "built": [1000.0, 100.0, 100.0]})) != [])
    check("declared_vs_built not PASS is refused", planted("FTRSP_s0", lambda sc: sc.update(declared_vs_built="FAIL")) != [])
    check("wrong wm parameter count (a lever that adds a parameter) is refused", planted("FTRSP_s0", lambda sc: sc["levers"]["wm_only"].update(n_wm=CK.WM_PARAMS_BASE + 1)) != [])
    # Option B levers
    check("wm.proj trainable in the BUILT model is refused", planted("FTRSP_s0", lambda sc: rb(sc, wm_proj_requires_grad=[True])) != [])
    check("mixed wm.proj requires_grad is refused", planted("FTRSP_s0", lambda sc: rb(sc, wm_proj_requires_grad=[False, True])) != [])
    check("freeze declared off is refused", planted("FTNULL_s0", lambda sc: rb(sc, freeze_proj_declared="off")) != [])
    check("init-from-ckpt-wm declared off is refused", planted("FTRSP_s0", lambda sc: rb(sc, init_from_ckpt_wm_declared="off")) != [])
    check("built readout idm on an RSP arm is refused", planted("FTRSP_s0", lambda sc: rb(sc, built_readout="idm")) != [])
    check("built slope off by 1e-3 is refused", planted("FTRSP_s0", lambda sc: sc["levers"]["wm_rsp_readback"]["built"].update(s_cal=CK.S_CAL[0] + 1e-3)) != [])
    check("trainable count equal to the full WM (nothing frozen) is refused", planted("FTRSP_s0", lambda sc: sc["levers"]["wm_only"].update(n_trainable=CK.WM_PARAMS_BASE)) != [])
    check("a trainable count that differs between the reads is refused", planted("FTRSP_s0", lambda sc: sc["levers"].update(n_params_trainable=CK.WM_PARAMS_BASE - 1)) != [])
    check("wm_only record from the old re-init mode is refused", planted("FTRSP_s0", lambda sc: sc["info"]["wm_only"].update(mode="wm_only", init_wm=False)) != [])
    check("a wm_reinit_seed in the record is refused", planted("FTRSP_s0", lambda sc: sc["info"]["wm_only"].update(wm_reinit_seed=0)) != [])
    check("the init checkpoint path of the wrong chassis is refused", planted("FTRSP_s0", lambda sc: sc["info"]["wm_only"].update(path=sc["info"]["wm_only"]["path"].replace("s0", "s1"))) != [])
    check("a wrong init sha256 is refused", planted("FTRSP_s0", lambda sc: sc["info"]["wm_only"].update(sha256="0" * 64)) != [])
    check("chassis-1 arm accepts its own slope and refuses chassis 0's", planted("FTRSP_s1", lambda sc: None) == [] and planted("FTRSP_s1", lambda sc: sc["levers"]["wm_rsp_readback"]["built"].update(s_cal=CK.S_CAL[0])) != [])
    # the log identity (card C2)
    def rows(rsp_w, with_key=True, gamma1=100.0):
        out = []
        for s in (50, 100, 150):
            r = {"step": s}
            for L in (1, 2, 3):
                pred, sig, roll, asg, idm = 0.2, 3.0, 0.25, 8.0 if L >= 2 else 0.0, 0.11 * L
                r.update({f"wm/pred/L{L}": pred, f"wm/sigreg/L{L}": sig, f"wm/rollout/L{L}": roll, f"wm/idm/L{L}": idm})
                if L >= 2:
                    r[f"wm/act_sigreg/L{L}"] = asg
                r[f"wm/level/L{L}"] = pred + roll + 0.08 * sig + (gamma1 if L == 1 else 100.0) * idm + (0.005 * asg if L >= 2 else 0.0)
            if with_key:
                r["wm/rsp/L1"] = 0.4 + 0.01 * s / 50
                r["wm/level/L1"] += rsp_w * r["wm/rsp/L1"]
            out.append(r)
        return out
    tmp = tempfile.mkdtemp(prefix="ckrsp_", dir=os.environ.get("TMP"))

    def log(rs):
        p = os.path.join(tmp, "train_log.jsonl")
        open(p, "w", encoding="utf-8").write("\n".join(json.dumps(r) for r in rs) + "\n")
        return p
    check("RSP log with rsp_w 1.0 closes", CK.gamma_log_problems("FTRSP_s0", log(rows(1.0)))[1] == [])
    check("RSP log with rsp_w 0.0 (term logged, not added) is refused", CK.gamma_log_problems("FTRSP_s0", log(rows(0.0)))[1] != [])
    check("RSP log without the rsp key is refused", CK.gamma_log_problems("FTRSP_s0", log(rows(0.0, with_key=False)))[1] != [])
    check("NULL log without the key closes", CK.gamma_log_problems("FTNULL_s4", log(rows(0.0, with_key=False)))[1] == [])
    check("NULL log carrying the rsp key is refused", CK.gamma_log_problems("FTNULL_s4", log(rows(1.0)))[1] != [])
    check("the token schema stays actsens-gate-token-v1", "actsens-gate-token-v1" in open(os.path.join(PKG, "optb_check.py"), encoding="utf-8").read())
    print(f"UNIT ok={N_OK} bad={N_BAD}")
    return 1 if N_BAD else 0


if __name__ == "__main__":
    sys.exit(main())
